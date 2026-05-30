from crewai import Task
from agents import coder, reviewer, tester, fixer, integrator

# ── Shared output format instruction ─────────────────────────────────────────
_FILE_BLOCK_FORMAT = (
    "Output every file in a clearly labelled block using this EXACT format:\n"
    "--- filename.py ---\n"
    "<file contents>\n"
    "--- other_filename.py ---\n"
    "<file contents>\n"
    "--- datafile.csv ---\n"
    "<file contents>\n\n"
    "This applies to ALL file types — Python files, CSV files, JSON files, etc. "
    "Use the exact filenames the task specifies. "
    "Do not merge multiple files into one block. "
    "Do not add any text outside the labelled blocks."
)


def build_coder_task(feature_request: str) -> Task:
    """Phase 1: write the implementation."""
    return Task(
        description=(
            "/no_think\n"
            f"Implement the following in Python:\n\n{feature_request}\n\n"
            + _FILE_BLOCK_FORMAT + "\n\n"
            "General rules that apply to ALL implementations:\n"
            "- When removing unwanted characters from strings, replace them with spaces, "
            "never delete them\n"
            "- Always handle None, empty string, and whitespace-only inputs explicitly\n"
            "- Never use mutable default arguments (e.g. def f(x=[]) is a bug)\n"
            "- Always think: what happens at the boundaries?\n"
            "- Use type hints on all function signatures\n"
            "- Use docstrings inside the code to explain your logic\n\n"
            "Architecture rules:\n"
            "- Each file named in the spec has a designated responsibility. "
            "Keep those responsibilities strictly separated — never migrate logic from one file to another.\n"
            "- If a file is designated for storage or I/O (e.g. storage.py), ALL file reading "
            "and writing must live exclusively in that file. "
            "Business logic files must not open, read, or write files directly.\n"
            "- The pipeline has three output folders that are filled automatically by routing "
            "each file block to the right place based on its name and extension:\n"
            "    src/   <- all .py implementation files (e.g. library.py, storage.py)\n"
            "    tests/ <- test files whose name starts with test_ (e.g. test_library.py)\n"
            "    data/  <- data files: .csv, .json, .yaml, .toml, .xml, .tsv\n"
            "  Output data files as bare-filename labelled blocks (e.g. '--- books.csv ---', "
            "NOT '--- data/books.csv ---') and they will land in data/ automatically. "
            "Never write code that creates data files at runtime instead of outputting them "
            "as file blocks — runtime creation puts them in the wrong place during tests.\n\n"
            "OUTPUT RULE: Respond with ONLY the labelled file blocks. "
            "No introduction, no explanation, no summary — just the file blocks."
        ),
        expected_output=(
            "All required Python files, each in a '--- filename.py ---' labelled block, "
            "with type hints, docstrings, and no placeholder TODOs. "
            "No text outside the labelled blocks."
        ),
        agent=coder,
    )


def build_edit_task(edit_instruction: str, existing_files: dict) -> Task:
    """Phase 1 for edit mode: apply a targeted change to existing files."""
    files_text = "\n\n".join(
        "--- " + fname + " ---\n```python\n" + code + "\n```"
        for fname, code in existing_files.items()
    )
    return Task(
        description=(
            "/no_think\n"
            "You are editing existing Python files. Apply the requested change precisely.\n\n"
            "Current files:\n\n" + files_text + "\n\n"
            "Edit instruction: " + edit_instruction + "\n\n"
            "Rules:\n"
            "- Output ALL files (both modified and unmodified) in labelled blocks\n"
            "- Make only the changes needed to implement the instruction\n"
            "- Preserve all existing functionality not mentioned in the instruction\n"
            "- Use type hints, docstrings, and follow PEP8\n"
            "- Respect existing file responsibilities — do not move logic between files\n"
            "- If a storage/IO file exists, all file reading and writing must stay there\n"
            "- Data files (CSV, JSON, etc.) must be output as bare-filename labelled blocks "
            "(e.g. '--- books.csv ---'), never as runtime file-creation code\n\n"
            + _FILE_BLOCK_FORMAT + "\n\n"
            "OUTPUT RULE: Respond with ONLY the labelled file blocks. No explanation."
        ),
        expected_output=(
            "All project files in '--- filename.py ---' labelled blocks, "
            "with the requested edit applied and all other files unchanged."
        ),
        agent=coder,
    )


def build_remaining_tasks(
    feature_request: str,
    code_task: Task,
    real_outputs: str = "",
) -> list[Task]:
    """
    Phase 2: review -> test -> fix -> integrate.

    real_outputs: newline-separated 'repr(input)  ->  repr(output)' pairs produced
                  by actually running the implementation (empty string if sampler failed).
    """

    review_task = Task(
        description=(
            "Review the Python code produced by the developer.\n\n"
            "Check for:\n"
            "1. Correctness - does it actually solve the task?\n"
            "2. Edge cases - what inputs could break it?\n"
            "3. Readability - is it clear and well-structured?\n"
            "4. Best practices - PEP8, error handling, type hints\n\n"
            "Always check these common Python bugs regardless of task:\n"
            "- String stripping/replacement: does removing characters accidentally merge tokens?\n"
            "- Off-by-one errors in index or range operations\n"
            "- Mutable default arguments\n"
            "- Missing None/empty input handling\n"
            "- Functions that silently return None instead of raising on invalid input\n"
            "- Incorrect use of is vs == for comparisons\n\n"
            "List your findings as numbered points. "
            "Reference function names and file names. "
            "End with a one-line verdict: APPROVED or NEEDS CHANGES."
        ),
        expected_output=(
            "A numbered list of review findings ending with APPROVED or NEEDS CHANGES."
        ),
        agent=reviewer,
        context=[code_task],
    )

    # ── Tester prompt: Option A (ground-truth) + Option C (property tests) ────
    if real_outputs:
        outputs_section = (
            "GROUND-TRUTH OUTPUTS — produced by running the actual implementation.\n"
            "These are facts, not guesses. Use them exactly as shown.\n\n"
            + real_outputs + "\n\n"
            "Write TWO kinds of tests:\n"
            "1. CONCRETE TESTS — one test per ground-truth pair above. "
            "Assert the exact output shown; copy the value literally, do not recompute it.\n"
            "2. PROPERTY TESTS — assert structural invariants that must hold for any input "
            "(e.g. return type is dict, all keys are strings, all values are ints > 0, "
            "no key contains a character that the spec says to replace). "
            "These must NOT hardcode specific expected values.\n\n"
            "RULE: Never invent or guess an expected value. "
            "Only use ground-truth values for concrete assertions.\n\n"
        )
    else:
        outputs_section = (
            "RULE: Derive every expected value by tracing the implementation line by line "
            "on the exact input — never guess. "
            "For any case where the expected value is ambiguous, write a property test "
            "(checking return type, key/value types, invariants) instead of a concrete assertion.\n\n"
        )

    test_task = Task(
        description=(
            "/no_think\n"
            "Write a pytest test suite for the Python code.\n\n"
            "IMPORTANT: Only test functions and behaviours that exist in the code. "
            "Do not invent methods or validation that are not present.\n\n"
            + outputs_section
            + "Requirements:\n"
            "- The project has a fixed directory layout:\n"
            "    src/   <- all implementation .py files live here\n"
            "    tests/ <- all test files live here (you are writing into tests/)\n"
            "    data/  <- data files (.csv, .json, etc.) live here\n"
            "  The test runner is configured with 'pythonpath = src', so src/ is already "
            "on sys.path. Import modules by their bare name exactly as they appear in src/.\n"
            "  For example: 'from library import LibraryManager' or 'from storage import load'. "
            "Never add sys.path manipulation, never import from a module that does not exist "
            "in the coder's output.\n"
            "- Add 'import pytest' at the top\n"
            "- Use pytest fixtures where appropriate\n"
            "- Use descriptive test function names\n"
            "- Tests must be self-contained with no external dependencies\n\n"
            "Name the test file to match the project: e.g. test_todo.py, test_cache.py.\n"
            + _FILE_BLOCK_FORMAT + "\n\n"
            "OUTPUT RULE: Respond with ONLY the labelled test file block. "
            "No introduction, no explanation, no summary — just the file block."
        ),
        expected_output=(
            "A complete test file in a '--- test_*.py ---' labelled block, "
            "with correct imports, ready to run with pytest. "
            "No text outside the labelled block."
        ),
        agent=tester,
        context=[code_task, review_task],
    )

    fix_task = Task(
        description=(
            "/no_think\n"
            "You have: the original code files, the code review, and the test suite.\n\n"
            "Your job:\n"
            "1. Fix every issue raised in the review\n"
            "2. Make sure the code would pass all the tests\n"
            "3. Check that every import in the test file matches exactly what is defined "
            "in the implementation files — fix any name mismatches or missing imports\n"
            "4. Do not remove any existing functionality\n"
            "5. If the code removes characters, replace with spaces, never delete\n\n"
            "IMPORTANT: Only output files you actually modified. "
            "If a file is already correct, omit it — it is already saved to disk. "
            "Do not re-output unchanged files.\n\n"
            + _FILE_BLOCK_FORMAT
        ),
        expected_output=(
            "Only the modified project files in labelled '--- filename.py ---' blocks. "
            "Unchanged files are omitted."
        ),
        agent=fixer,
        context=[code_task, review_task, test_task],
    )

    return [review_task, test_task, fix_task]


def build_repair_tasks(
    current_files: dict,
    pytest_output: str,
    cycle: int,
    feature_request: str = "",
) -> list[Task]:
    """
    2-task repair pipeline for the executor loop.
    Fixer receives ALL current project files + the failing pytest output.
    If feature_request is provided, the fixer may also correct a wrong test
    whose expected value contradicts the original specification.
    """

    files_text = "\n\n".join(
        "--- " + fname + " ---\n```python\n" + code + "\n```"
        for fname, code in current_files.items()
    )

    test_files = [f for f in current_files if f.startswith("test_") or f.endswith("_test.py")]

    if feature_request:
        spec_block = (
            "ORIGINAL SPECIFICATION (source of truth):\n"
            "```\n" + feature_request + "\n```\n\n"
        )
        test_rule = (
            "Priority rule: if a test's expected value contradicts the original "
            "specification above, fix the test — not the implementation. "
            "Only fix tests that are demonstrably wrong per the spec; "
            "do not change any other test logic."
        )
    else:
        spec_block = ""
        test_rule = (
            "Fix only the implementation files so every test passes. "
            "Do not modify test files unless a test is demonstrably wrong."
        )

    fix_task = Task(
        description=(
            "/no_think\n"
            "The test suite is FAILING. This is repair cycle " + str(cycle) + ".\n\n"
            + spec_block
            + "Here are ALL current project files:\n\n" + files_text + "\n\n"
            "Here is the pytest output showing exactly what failed:\n"
            "```\n" + pytest_output + "\n```\n\n"
            + test_rule + "\n\n"
            "Trace through the failing assertion line by line before writing your fix.\n\n"
            + _FILE_BLOCK_FORMAT + "\n\n"
            "OUTPUT RULE: Respond with ONLY the labelled file blocks. No explanation."
        ),
        expected_output=(
            "All corrected project files in labelled '--- filename.py ---' blocks."
        ),
        agent=fixer,
    )

    integrate_task = Task(
        description=(
            "/no_think\n"
            "You have repaired project files and the test suite.\n\n"
            "1. Verify every import in the test file(s) matches exactly what is defined "
            "in the implementation files\n"
            "2. Fix any name mismatches or missing imports\n"
            "3. Do not change any test logic\n\n"
            + _FILE_BLOCK_FORMAT + "\n\n"
            "OUTPUT RULE: Respond with ONLY the labelled file blocks. No explanation."
        ),
        expected_output=(
            "ALL project files in labelled '--- filename.py ---' blocks, "
            "complete and ready to run with pytest."
        ),
        agent=integrator,
        context=[fix_task],
    )

    return [fix_task, integrate_task]
