from crewai import Task
from agents import coder, reviewer, tester, fixer, integrator

# ── Shared output format instruction ─────────────────────────────────────────
_FILE_BLOCK_FORMAT = (
    "Output every file in a clearly labelled block using this EXACT format:\n"
    "--- filename.py ---\n"
    "<file contents>\n"
    "--- other_filename.py ---\n"
    "<file contents>\n\n"
    "Use the exact filenames the task specifies. "
    "Do not merge multiple files into one block. "
    "Do not add any text outside the labelled blocks."
)


def build_tasks(feature_request: str) -> list[Task]:
    """
    5-task pipeline: code -> review -> test -> fix -> integrate.
    Supports any number of output files with any names.
    """

    code_task = Task(
        description=(
            f"Implement the following in Python:\n\n{feature_request}\n\n"
            + _FILE_BLOCK_FORMAT + "\n\n"
            "General rules that apply to ALL implementations:\n"
            "- When removing unwanted characters from strings, replace them with spaces, "
            "never delete them\n"
            "- Always handle None, empty string, and whitespace-only inputs explicitly\n"
            "- Never use mutable default arguments (e.g. def f(x=[]) is a bug)\n"
            "- Always think: what happens at the boundaries?\n"
            "- Use type hints on all function signatures\n"
            "- Use docstrings inside the code to explain your logic\n"
        ),
        expected_output=(
            "All required Python files, each in a '--- filename.py ---' labelled block, "
            "with type hints, docstrings, and no placeholder TODOs."
        ),
        agent=coder,
    )

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

    test_task = Task(
        description=(
            "Write a pytest test suite for the Python code.\n\n"
            "IMPORTANT: Only test functions and behaviours that exist in the code. "
            "Do not invent methods or validation that are not present.\n\n"
            "Requirements:\n"
            "- Import from the correct module(s) based on the project file structure. "
            "Use the actual filenames (without .py) as module names. "
            "For example: 'from todo import TodoList' or 'from storage import load_todos'. "
            "Never import from a module that does not exist in the coder's output.\n"
            "- Add 'import pytest' at the top\n"
            "- Use pytest fixtures where appropriate\n"
            "- Use descriptive test function names\n"
            "- Tests must be self-contained with no external dependencies\n\n"
            "Every test suite must include: happy path, None input, empty input, "
            "boundary cases, and edge cases flagged in the review.\n\n"
            "Name the test file to match the project: e.g. test_todo.py, test_cache.py.\n"
            + _FILE_BLOCK_FORMAT
        ),
        expected_output=(
            "A complete test file in a '--- test_*.py ---' labelled block, "
            "with correct imports, ready to run with pytest."
        ),
        agent=tester,
        context=[code_task, review_task],
    )

    fix_task = Task(
        description=(
            "You have: the original code files, the code review, and the test suite.\n\n"
            "Your job:\n"
            "1. Fix every issue raised in the review\n"
            "2. Make sure the code would pass all the tests\n"
            "3. Do not remove any existing functionality\n"
            "4. If the code removes characters, replace with spaces, never delete\n\n"
            + _FILE_BLOCK_FORMAT
        ),
        expected_output=(
            "All corrected project files in labelled '--- filename.py ---' blocks."
        ),
        agent=fixer,
        context=[code_task, review_task, test_task],
    )

    integrate_task = Task(
        description=(
            "You have the final implementation files and the test suite.\n\n"
            "Your job:\n"
            "1. Check that every import in the test file matches exactly what is defined "
            "in the implementation files\n"
            "2. Check that every function/class called in the tests exists in the "
            "implementation with the exact same name\n"
            "3. Fix any mismatches - wrong import names, missing imports, typos\n"
            "4. Do not change any test logic\n\n"
            + _FILE_BLOCK_FORMAT
        ),
        expected_output=(
            "ALL project files in labelled '--- filename.py ---' blocks, "
            "complete and ready to run with pytest."
        ),
        agent=integrator,
        context=[fix_task, test_task],
    )

    return [code_task, review_task, test_task, fix_task, integrate_task]


def build_repair_tasks(current_files: dict, pytest_output: str, cycle: int) -> list[Task]:
    """
    2-task repair pipeline for the executor loop.
    Fixer receives ALL current project files + the failing pytest output.
    """

    files_text = "\n\n".join(
        "--- " + fname + " ---\n```python\n" + code + "\n```"
        for fname, code in current_files.items()
    )

    test_files = [f for f in current_files if f.startswith("test_") or f.endswith("_test.py")]
    if test_files:
        test_note = "The test file(s) are: " + ", ".join(test_files) + ". DO NOT MODIFY THEM."
    else:
        test_note = "Do not modify any test files."

    fix_task = Task(
        description=(
            "The test suite is FAILING. This is repair cycle " + str(cycle) + ".\n\n"
            "Here are ALL current project files:\n\n" + files_text + "\n\n"
            "Here is the pytest output showing exactly what failed:\n"
            "```\n" + pytest_output + "\n```\n\n"
            + test_note + "\n"
            "Fix only the implementation files so every test passes. "
            "Trace through the logic line by line before writing your fix.\n\n"
            + _FILE_BLOCK_FORMAT
        ),
        expected_output=(
            "All corrected project files in labelled '--- filename.py ---' blocks. "
            "Test files must be identical to the input."
        ),
        agent=fixer,
    )

    integrate_task = Task(
        description=(
            "You have repaired project files and the test suite.\n\n"
            "1. Verify every import in the test file(s) matches exactly what is defined "
            "in the implementation files\n"
            "2. Fix any name mismatches or missing imports\n"
            "3. Do not change any test logic\n\n"
            + _FILE_BLOCK_FORMAT
        ),
        expected_output=(
            "ALL project files in labelled '--- filename.py ---' blocks, "
            "complete and ready to run with pytest."
        ),
        agent=integrator,
        context=[fix_task],
    )

    return [fix_task, integrate_task]
