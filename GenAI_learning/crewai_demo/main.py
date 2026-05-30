#!/usr/bin/env python3

# -- Default task ---------------------------------------------------------------

DEFAULT_TASK = (
    "Create a CLI todo app with add, list, done, delete commands. "
    "Split into main.py, todo.py, and storage.py."
)

"""
Multi-agent coding crew using CrewAI + Ollama.

Pipeline:
  1. Coder       -> writes the code
  2. Sampler     -> runs the code on LLM-generated inputs, collects real outputs
  3. Reviewer    -> reviews the code
  4. Tester      -> writes tests using real ground-truth outputs + property tests
  5. Fixer       -> fixes code based on review + tests
  6. Integrator  -> checks imports and wiring across all files
  7. Executor    -> runs pytest; loops back to fixer if failing (max 2 cycles)

Output layout:
  output/src/    - implementation .py files
  output/tests/  - test_*.py / *_test.py files
  output/data/   - csv, json, yaml, etc.
  output/review.txt

Usage:
  python main.py
  python main.py "Write a function that counts word frequency in a string"
"""
import argparse
import ast
import json
import os
import re
import shutil
import sys
import time
import subprocess
from crewai import Crew, Process
from agents import coder, reviewer, tester, fixer, integrator, fast_llm
from tasks import build_coder_task, build_edit_task, build_remaining_tasks, build_repair_tasks

MAX_REPAIR_CYCLES = 2

# Subdirectory names inside output/
SRC_DIR   = "src"
TESTS_DIR = "tests"
DATA_DIR  = "data"

# File extensions routed to data/
DATA_EXTENSIONS = {".csv", ".json", ".yaml", ".yml", ".toml", ".xml", ".tsv"}


# -- File classification -------------------------------------------------------

def classify_file(filename: str) -> str:
    """Return the output subdirectory for a given filename."""
    name = filename.lower()
    ext  = os.path.splitext(name)[1]
    if name.startswith("test_") or name.endswith("_test.py"):
        return TESTS_DIR
    if ext in DATA_EXTENSIONS:
        return DATA_DIR
    return SRC_DIR


# -- File I/O helpers -----------------------------------------------------------

def clear_output(output_dir: str) -> None:
    """Delete all .py files from output/src/ and output/tests/ before a new run."""
    for subdir in [SRC_DIR, TESTS_DIR]:
        target = os.path.join(output_dir, subdir)
        if not os.path.isdir(target):
            continue
        for fname in os.listdir(target):
            if fname.endswith(".py") and not fname.startswith("__"):
                os.remove(os.path.join(target, fname))


def extract_code(text: str) -> str:
    """Strip markdown code fences if the model wrapped output in them."""
    text = str(text)
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def parse_all_blocks(text: str) -> dict:
    """
    Scan for every '--- filename.ext ---' block.
    Returns {filename: content} for any number of files with any names/extensions.
    """
    text    = str(text)
    pattern = r"---\s*([\w\-]+\.[\w]+)\s*---\s*(.*?)(?=---\s*[\w\-]+\.[\w]+\s*---|$)"
    result  = {}
    for m in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
        filename = m.group(1).strip()
        content  = extract_code(m.group(2).strip())
        if content:
            result[filename] = content
    return result


def save_all_files(files: dict, output_dir: str) -> list:
    """
    Write each file to its classified subfolder inside output_dir.
    Returns list of saved absolute paths.
    """
    saved = []
    for filename, content in files.items():
        subdir     = classify_file(filename)
        target_dir = os.path.join(output_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  Saved: output/" + subdir + "/" + filename)
        saved.append(path)
    return saved


def read_output_files(output_dir: str) -> dict:
    """Read all .py files from src/ and tests/ for use in repair prompts."""
    files = {}
    for subdir in [SRC_DIR, TESTS_DIR]:
        target = os.path.join(output_dir, subdir)
        if not os.path.isdir(target):
            continue
        for fname in sorted(os.listdir(target)):
            if fname.endswith(".py") and not fname.startswith("__"):
                with open(os.path.join(target, fname), encoding="utf-8") as f:
                    files[fname] = f.read()
    return files


# -- Sampler -------------------------------------------------------------------

def run_sampler(files: dict, feature_request: str, output_dir: str) -> str:
    """
    Execute the coder's implementation on LLM-generated inputs and return
    real 'repr(input)  ->  repr(output)' pairs for the tester to use as
    ground truth (Option A).

    Returns an empty string if anything fails — the tester degrades gracefully.
    """
    # Only use implementation files (not tests)
    impl_files = {
        f: c for f, c in files.items()
        if not (f.startswith("test_") or f.endswith("_test.py"))
    }
    if not impl_files:
        return ""

    # ── Find the main public function via AST ─────────────────────────────────
    module_name = None
    func_name   = None
    for fname, code in impl_files.items():
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    module_name = fname.replace(".py", "")
                    func_name   = node.name
                    break
        except Exception:
            continue
        if func_name:
            break

    if not module_name or not func_name:
        return ""

    # ── Ask the LLM for representative inputs ─────────────────────────────────
    impl_text = "\n\n".join(f"# {f}\n{c}" for f, c in impl_files.items())
    prompt = (
        "/no_think\n"
        f"Function to test: {module_name}.{func_name}\n\n"
        f"Spec:\n{feature_request}\n\n"
        f"Implementation:\n{impl_text}\n\n"
        "Generate 8-10 representative test inputs for this function covering "
        "happy paths, boundary cases, and edge cases from the spec. "
        "Respond with ONLY a JSON array of inputs. Use null for None. "
        "Example: [null, \"\", \"hello world\"]\n"
        "JSON array:"
    )

    try:
        raw    = fast_llm.call(messages=[{"role": "user", "content": prompt}])
        arr_m  = re.search(r"\[.*?\]", str(raw), re.DOTALL)
        if not arr_m:
            return ""
        inputs = json.loads(arr_m.group())
        if not isinstance(inputs, list) or not inputs:
            return ""
    except Exception:
        return ""

    # ── Save implementation to a temp dir and execute ─────────────────────────
    src_dir = os.path.join(output_dir, "_sampler_src")
    os.makedirs(src_dir, exist_ok=True)
    try:
        for fname, code in impl_files.items():
            with open(os.path.join(src_dir, fname), "w", encoding="utf-8") as fh:
                fh.write(code)

        exec_script = (
            "import sys, json\n"
            f"sys.path.insert(0, {repr(src_dir)})\n"
            f"from {module_name} import {func_name}\n"
            "inputs = json.loads(sys.stdin.read())\n"
            "results = []\n"
            "for inp in inputs:\n"
            "    try:\n"
            f"        out = {func_name}(inp)\n"
            "        results.append({'i': inp, 'o': out, 'e': None})\n"
            "    except Exception as ex:\n"
            "        results.append({'i': inp, 'o': None, 'e': str(ex)})\n"
            "print(json.dumps(results, default=str))\n"
        )

        try:
            proc = subprocess.run(
                [sys.executable, "-c", exec_script],
                input=json.dumps(inputs),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return ""
            results = json.loads(proc.stdout.strip())
        except Exception:
            return ""
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)

    # ── Format as readable pairs ──────────────────────────────────────────────
    lines = []
    for r in results:
        if r["e"]:
            lines.append(f"  {repr(r['i'])}  ->  ERROR: {r['e']}")
        else:
            lines.append(f"  {repr(r['i'])}  ->  {repr(r['o'])}")
    return "\n".join(lines)


# -- Executor -------------------------------------------------------------------

def has_test_files(output_dir: str) -> bool:
    """Return True if output/tests/ contains at least one test_*.py file."""
    tests_dir = os.path.join(output_dir, TESTS_DIR)
    if not os.path.isdir(tests_dir):
        return False
    return any(
        f.endswith(".py") and (f.startswith("test_") or f.endswith("_test.py"))
        for f in os.listdir(tests_dir)
    )


def run_pytest(output_dir: str):
    """
    Run pytest from output/ using pytest.ini (pythonpath = src, testpaths = tests).
    PYTHONPATH is also set as a belt-and-suspenders fallback.
    Returns (all_passed, output_text).
    """
    if not has_test_files(output_dir):
        return False, "No test file found in output/tests/ (expected test_*.py or *_test.py)"

    src_dir = os.path.join(output_dir, SRC_DIR)
    env     = dict(os.environ)
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--tb=short", "-v", "--no-header"],
        capture_output=True,
        text=True,
        cwd=output_dir,   # run from output/ so pytest.ini is picked up
        env=env,
    )
    output = result.stdout
    if result.stderr.strip():
        output += "\n" + result.stderr
    return result.returncode == 0, output


# -- Mutable stdout wrapper ----------------------------------------------------

class _MutableStdout:
    """Thin wrapper around the real stdout that can be muted on demand."""
    def __init__(self, real):
        self._real = real
        self.muted = False
    def write(self, s):
        if not self.muted:
            self._real.write(s)
    def flush(self):
        self._real.flush()
    def fileno(self):
        return self._real.fileno()
    def isatty(self):
        return hasattr(self._real, "isatty") and self._real.isatty()

_real_out        = sys.__stdout__
_mutable_stdout  = _MutableStdout(_real_out)
sys.stdout       = _mutable_stdout

# Steps whose LLM output we want to hide (0=coder, 3=tester)
_SILENT_STEPS = {0, 3}


# -- Progress tracker -----------------------------------------------------------

STEP_LABELS = [
    ("Step 1/5", "Coder",    "writing code..."),
    ("Step 2/5", "Sampler",  "running code to collect ground-truth outputs..."),
    ("Step 3/5", "Reviewer", "reviewing code..."),
    ("Step 4/5", "Tester",   "writing tests..."),
    ("Step 5/5", "Fixer",    "fixing code and verifying imports..."),
]
_current_step  = [-1]
_step_start    = [0.0]
_total_start   = [0.0]


def _fmt_elapsed(seconds: float) -> str:
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    return f"{seconds:.1f}s"


def _print(msg: str = ""):
    """Write directly to the real terminal, bypassing mute state."""
    _real_out.write(msg + "\n")
    _real_out.flush()


def print_step_start(step: int):
    if 0 <= step < len(STEP_LABELS):
        icon, role, action = STEP_LABELS[step]
        _step_start[0] = time.time()
        _print(f"\n{icon}  {role} is {action}")
        _mutable_stdout.muted = step in _SILENT_STEPS


def on_task_complete(task_output):
    _mutable_stdout.muted = False
    step = _current_step[0]
    if 0 <= step < len(STEP_LABELS):
        _, role, _ = STEP_LABELS[step]
        elapsed = time.time() - _step_start[0]
        _print(f"   ✓ {role} done  [{_fmt_elapsed(elapsed)}]")
    _current_step[0] += 1
    print_step_start(_current_step[0])


# -- Pipeline helpers ----------------------------------------------------------

def _repair_loop(output_dir: str, feature_request: str) -> bool:
    """
    Run pytest; if failing, attempt up to MAX_REPAIR_CYCLES repair cycles.
    Returns True if all tests pass at the end.
    """
    passed, pytest_out = run_pytest(output_dir)
    print(pytest_out)

    if passed:
        total = _fmt_elapsed(time.time() - _total_start[0])
        _print(f"All tests passed — no repairs needed.  [total: {total}]")
        return True

    for cycle in range(1, MAX_REPAIR_CYCLES + 1):
        print("\n" + "=" * 60)
        print("Repair cycle " + str(cycle) + "/" + str(MAX_REPAIR_CYCLES))
        print("=" * 60)

        current_files = read_output_files(output_dir)
        repair_tasks  = build_repair_tasks(current_files, pytest_out, cycle, feature_request)

        print("\nRepair " + str(cycle) + " — Fixer is patching code...")
        repair_crew = Crew(
            agents=[fixer, integrator],
            tasks=repair_tasks,
            process=Process.sequential,
            verbose=False,
        )
        repair_crew.kickoff()

        repair_outputs    = [t.output.raw if t.output else "" for t in repair_tasks]
        repair_integrator = str(repair_outputs[-1])

        print("Repair " + str(cycle) + " — Integrator verifying wiring...")
        repaired = parse_all_blocks(repair_integrator)
        if not repaired:
            repaired = parse_all_blocks(repair_outputs[0])
        if repaired:
            save_all_files(repaired, output_dir)
        else:
            print("Warning: repair produced no labelled blocks — files unchanged.")

        print("\nExecutor: re-running pytest after repair " + str(cycle) + "...")
        passed, pytest_out = run_pytest(output_dir)
        print(pytest_out)

        if passed:
            total = _fmt_elapsed(time.time() - _total_start[0])
            _print(f"All tests passed after repair cycle {cycle}!  [total: {total}]")
            return True

    total = _fmt_elapsed(time.time() - _total_start[0])
    _print(f"Tests still failing after {MAX_REPAIR_CYCLES} repair cycles.  [total: {total}]")
    _print("Final pytest output above. Check output/ for the latest code.")
    return False


def _save_phase2_output(remaining_tasks: list, output_dir: str) -> None:
    """Save tester + fixer output files and review.txt.

    Save order:
      1. Tester output  — always saved (test file is new every run)
      2. Fixer output   — only changed files, overwrites coder draft or tester file if modified
    """
    # remaining_tasks = [review(0), test(1), fix(2)]
    r_outputs = [t.output.raw if t.output else "" for t in remaining_tasks]

    print()

    # 1. Always save the tester's test file(s)
    test_files = parse_all_blocks(str(r_outputs[1]))
    if test_files:
        save_all_files(test_files, output_dir)
    else:
        print("  Warning: tester produced no labelled test file block.")

    # 2. Save only fixer-modified files (overwrites coder draft or tester output if changed)
    fix_files = parse_all_blocks(str(r_outputs[2]))
    if fix_files:
        save_all_files(fix_files, output_dir)
    else:
        print("  Note: fixer made no changes — coder draft already on disk.")

    review_path = os.path.join(output_dir, "review.txt")
    with open(review_path, "w", encoding="utf-8") as f:
        f.write(str(r_outputs[0]))
    print("  Saved: output/review.txt")


def _run_phase1_and_sampler(code_task, feature_request: str, output_dir: str) -> tuple:
    """Run the coder crew and sampler; return (real_outputs, coder_files)."""
    phase1_crew = Crew(
        agents=[coder],
        tasks=[code_task],
        process=Process.sequential,
        verbose=False,
        task_callback=on_task_complete,
    )
    _current_step[0] = 0
    print_step_start(0)
    phase1_crew.kickoff()
    _mutable_stdout.muted = False

    sampler_start = time.time()
    coder_raw    = str(code_task.output.raw if code_task.output else "")
    coder_files  = parse_all_blocks(coder_raw)
    real_outputs = run_sampler(coder_files, feature_request, output_dir)
    _print(f"   ✓ Sampler done  [{_fmt_elapsed(time.time() - sampler_start)}]"
           + ("" if real_outputs else "  (no outputs — tester will use property tests only)"))
    return real_outputs, coder_files


def _run_phase2(code_task, feature_request: str, real_outputs: str, output_dir: str) -> list:
    """Run reviewer → tester → fixer; return remaining_tasks list."""
    _current_step[0] = 2
    print_step_start(2)

    remaining_tasks = build_remaining_tasks(feature_request, code_task, real_outputs)
    phase2_crew = Crew(
        agents=[reviewer, tester, fixer],
        tasks=remaining_tasks,
        process=Process.sequential,
        verbose=False,
        task_callback=on_task_complete,
    )
    phase2_crew.kickoff()
    _mutable_stdout.muted = False
    return remaining_tasks


# -- Run modes -----------------------------------------------------------------

def run_full_pipeline(feature_request: str, output_dir: str) -> None:
    """Default mode: full pipeline from scratch."""
    print("\n" + "=" * 60)
    print("TASK:")
    print(feature_request.strip())
    print("=" * 60)

    clear_output(output_dir)
    print("\nStarting crew...\n")
    _total_start[0] = time.time()

    code_task              = build_coder_task(feature_request)
    real_outputs, coder_files = _run_phase1_and_sampler(code_task, feature_request, output_dir)
    save_all_files(coder_files, output_dir)      # save coder draft immediately
    remaining              = _run_phase2(code_task, feature_request, real_outputs, output_dir)
    _save_phase2_output(remaining, output_dir)   # overwrite only fixer-changed files

    print("\n" + "=" * 60)
    print("Step 6  Executor: running pytest...")
    print("=" * 60)
    _repair_loop(output_dir, feature_request)


def run_edit_mode(instruction: str, output_dir: str) -> None:
    """Edit mode: apply a targeted change to existing output/ files."""
    print("\n" + "=" * 60)
    print("EDIT:")
    print(instruction.strip())
    print("=" * 60)

    existing_files = read_output_files(output_dir)
    if not existing_files:
        _print("\nNo files found in output/src/ or output/tests/. Run without --edit first.")
        return

    _print(f"\nLoaded {len(existing_files)} existing file(s): {', '.join(existing_files)}")
    print("\nStarting edit crew...\n")
    _total_start[0] = time.time()

    code_task                 = build_edit_task(instruction, existing_files)
    real_outputs, coder_files = _run_phase1_and_sampler(code_task, instruction, output_dir)
    save_all_files(coder_files, output_dir)      # save edit draft immediately
    remaining                 = _run_phase2(code_task, instruction, real_outputs, output_dir)
    _save_phase2_output(remaining, output_dir)   # overwrite only fixer-changed files

    print("\n" + "=" * 60)
    print("Step 6  Executor: running pytest...")
    print("=" * 60)
    _repair_loop(output_dir, instruction)


def run_fix_mode(instruction: str, output_dir: str) -> None:
    """Fix mode: run pytest on existing files and repair if failing."""
    print("\n" + "=" * 60)
    print("FIX MODE" + (f": {instruction.strip()}" if instruction else ""))
    print("=" * 60)

    if not read_output_files(output_dir):
        _print("\nNo files found in output/src/ or output/tests/. Run without --fix first.")
        return

    _total_start[0] = time.time()
    print("\n" + "=" * 60)
    print("Executor: running pytest on existing files...")
    print("=" * 60)
    _repair_loop(output_dir, instruction)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CrewAI multi-agent coding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py "Build a URL shortener with Flask"\n'
            '  python main.py --edit "add a search command that filters by keyword"\n'
            '  python main.py --fix "fix the remaining errors in tests"\n'
            '  python main.py --fix'
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--edit", metavar="INSTRUCTION",
                       help="Apply an edit to existing output/ files")
    group.add_argument("--fix", metavar="INSTRUCTION", nargs="?", const="",
                       help="Repair failing tests in existing output/ files")
    parser.add_argument("task", nargs="?", default=None,
                        help="Task description for the full pipeline")
    args = parser.parse_args()

    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    # pytest.ini tells pytest to add src/ to sys.path and look for tests in tests/
    # This replaces conftest.py and works for both pipeline runs and manual pytest calls
    with open(os.path.join(output_dir, "pytest.ini"), "w", encoding="utf-8") as f:
        f.write("[pytest]\npythonpath = src\ntestpaths = tests\n")

    if args.fix is not None:
        run_fix_mode(args.fix, output_dir)
    elif args.edit:
        run_edit_mode(args.edit, output_dir)
    else:
        run_full_pipeline(args.task or DEFAULT_TASK, output_dir)


if __name__ == "__main__":
    main()
