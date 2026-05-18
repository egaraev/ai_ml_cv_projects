#!/usr/bin/env python3

# -- Default task ---------------------------------------------------------------

DEFAULT_TASK = (
    "Create a CLI todo app with add, list, done, delete commands. "
    "Split into main.py, todo.py, and storage.py. "
    "Write pytest tests in test_todo.py."
)

"""
Multi-agent coding crew using CrewAI + Ollama.

Pipeline:
  1. Coder       -> writes the code
  2. Reviewer    -> reviews the code
  3. Tester      -> writes pytest tests
  4. Fixer       -> fixes code based on review + tests
  5. Integrator  -> checks imports and wiring across all files
  6. Executor    -> runs pytest; loops back to fixer if failing (max 2 cycles)

Output layout:
  output/src/    - implementation .py files
  output/tests/  - test_*.py / *_test.py files
  output/data/   - csv, json, yaml, etc.
  output/review.txt

File names and quantities are driven entirely by the task - no hardcoding.

Usage:
  python main.py
  python main.py "Write a function that counts word frequency in a string"
"""
import os
import re
import sys
import subprocess
from crewai import Crew, Process
from agents import coder, reviewer, tester, fixer, integrator
from tasks import build_tasks, build_repair_tasks

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


# -- File parsing helpers -------------------------------------------------------

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


# -- File I/O helpers -----------------------------------------------------------

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


# -- Executor -------------------------------------------------------------------

def find_test_file(output_dir: str):
    """Return (tests_dir, filename) for the first test file found, or (None, None)."""
    tests_dir = os.path.join(output_dir, TESTS_DIR)
    if not os.path.isdir(tests_dir):
        return None, None
    for fname in sorted(os.listdir(tests_dir)):
        if fname.endswith(".py") and (fname.startswith("test_") or fname.endswith("_test.py")):
            return tests_dir, fname
    return None, None


def run_pytest(output_dir: str):
    """
    Run pytest from output/tests/ with PYTHONPATH pointing at output/src/.
    Returns (all_passed, output_text).
    """
    tests_dir, test_file = find_test_file(output_dir)
    if not test_file:
        return False, "No test file found in output/tests/ (expected test_*.py or *_test.py)"

    src_dir = os.path.join(output_dir, SRC_DIR)
    env     = dict(os.environ)
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "--tb=short", "-v", "--no-header"],
        capture_output=True,
        text=True,
        cwd=tests_dir,
        env=env,
    )
    output = result.stdout
    if result.stderr.strip():
        output += "\n" + result.stderr
    return result.returncode == 0, output


# -- Progress tracker -----------------------------------------------------------

STEP_LABELS = [
    ("Step 1/5", "Coder",      "writing code..."),
    ("Step 2/5", "Reviewer",   "reviewing code..."),
    ("Step 3/5", "Tester",     "writing tests..."),
    ("Step 4/5", "Fixer",      "fixing code..."),
    ("Step 5/5", "Integrator", "checking imports and wiring..."),
]
_current_step = [-1]


def print_step_start(step: int):
    if 0 <= step < len(STEP_LABELS):
        icon, role, action = STEP_LABELS[step]
        print("\n" + icon + "  " + role + " is " + action, flush=True)


def on_task_complete(task_output):
    step = _current_step[0]
    if 0 <= step < len(STEP_LABELS):
        _, role, _ = STEP_LABELS[step]
        print("   [done] " + role + " done.", flush=True)
    _current_step[0] += 1
    print_step_start(_current_step[0])





# -- Main -----------------------------------------------------------------------

def main():
    feature_request = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TASK

    print("\n" + "="*60)
    print("TASK:")
    print(feature_request.strip())
    print("="*60)

    tasks = build_tasks(feature_request)

    crew = Crew(
        agents=[coder, reviewer, tester, fixer, integrator],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        task_callback=on_task_complete,
    )

    print("\nStarting crew...\n")
    _current_step[0] = 0
    print_step_start(0)
    result = crew.kickoff()

    # -- Save initial outputs ---------------------------------------------------
    os.makedirs("output", exist_ok=True)
    output_dir = os.path.join(os.getcwd(), "output")

    outputs          = [t.output.raw if t.output else "" for t in tasks]
    integrator_out   = str(outputs[4] if len(outputs) > 4 else str(result))

    print()

    # Try integrator -> fixer -> single-file fallback
    files = parse_all_blocks(integrator_out)
    if not files:
        files = parse_all_blocks(str(outputs[3] if len(outputs) > 3 else ""))
    if not files:
        fallback = extract_code(str(outputs[3] if len(outputs) > 3 else str(result)))
        if fallback:
            files["implementation.py"] = fallback
            print("Warning: no labelled blocks found - saved fixer output as src/implementation.py")

    save_all_files(files, output_dir)

    review_path = os.path.join(output_dir, "review.txt")
    with open(review_path, "w", encoding="utf-8") as f:
        f.write(str(outputs[1] if len(outputs) > 1 else ""))
    print("  Saved: output/review.txt")

    # -- Executor loop ----------------------------------------------------------
    print("\n" + "="*60)
    print("Step 6  Executor: running pytest...")
    print("="*60)

    passed, pytest_out = run_pytest(output_dir)
    print(pytest_out)

    if passed:
        print("All tests passed on first run - no repairs needed.")
        return

    for cycle in range(1, MAX_REPAIR_CYCLES + 1):
        print("\n" + "="*60)
        print("Repair cycle " + str(cycle) + "/" + str(MAX_REPAIR_CYCLES))
        print("="*60)

        current_files = read_output_files(output_dir)
        repair_tasks  = build_repair_tasks(current_files, pytest_out, cycle)

        print("\nRepair " + str(cycle) + " - Fixer is patching code...")
        repair_crew = Crew(
            agents=[fixer, integrator],
            tasks=repair_tasks,
            process=Process.sequential,
            verbose=False,
        )
        repair_crew.kickoff()

        repair_outputs      = [t.output.raw if t.output else "" for t in repair_tasks]
        repair_integrator   = str(repair_outputs[-1])

        print("Repair " + str(cycle) + " - Integrator verifying wiring...")
        repaired = parse_all_blocks(repair_integrator)
        if not repaired:
            repaired = parse_all_blocks(repair_outputs[0])
        if repaired:
            save_all_files(repaired, output_dir)
        else:
            print("Warning: repair produced no labelled blocks - files unchanged.")

        print("\nExecutor: re-running pytest after repair " + str(cycle) + "...")
        passed, pytest_out = run_pytest(output_dir)
        print(pytest_out)

        if passed:
            print("All tests passed after repair cycle " + str(cycle) + "!")
            return

    print("Tests still failing after " + str(MAX_REPAIR_CYCLES) + " repair cycles.")
    print("Final pytest output above. Check output/ for the latest code.")


if __name__ == "__main__":
    main()
