from crewai import Agent, LLM

# ── LLMs ──────────────────────────────────────────────────────────────────────
# Using 9b for all agents so everything fits in VRAM and runs fast.
# Swap coder_llm to "ollama/qwen3-coder:30b" for higher quality (but slower).

BASE_MODEL = "ollama/qwen3-coder:30b"  # single model for all agents — no swap overhead

fast_llm = LLM(
    model=BASE_MODEL,                  # preferred: ollama/qwen3.5:9b
    base_url="http://localhost:11434",
)

coder_llm = LLM(
    model=BASE_MODEL,                  # preferred: ollama/qwen3-coder:30b
    base_url="http://localhost:11434",
)

tester_llm = LLM(
    model=BASE_MODEL,                  # preferred: ollama/qwen2.5-coder:7b
    base_url="http://localhost:11434",
)

# ── Agents ────────────────────────────────────────────────────────────────────

coder = Agent(
    role="Senior Python Developer",
    goal="Write clean, working, well-documented Python code that solves the task exactly.",
    backstory=(
        "You are an experienced Python developer who writes production-quality code. "
        "You always include docstrings, handle edge cases, and follow PEP8. "
        "When you receive review feedback, you incorporate it fully."
    ),
    llm=coder_llm,
    verbose=False,
)

reviewer = Agent(
    role="Code Reviewer",
    goal=(
        "Review Python code for correctness, clarity, edge cases, and best practices. "
        "Provide specific, actionable feedback."
    ),
    backstory=(
        "You are a meticulous code reviewer with a sharp eye for bugs, security issues, "
        "and design problems. You give concise, numbered feedback points. "
        "You do NOT rewrite the code — you only critique it."
    ),
    llm=fast_llm,
    verbose=False,
)

tester = Agent(
    role="QA Engineer",
    goal="Write comprehensive pytest tests that cover happy paths, edge cases, and error conditions.",
    backstory=(
        "You are a QA engineer who writes thorough test suites. "
        "You always use pytest fixtures, parametrize where appropriate, "
        "and make sure tests are isolated and repeatable. "
        "You read the implementation carefully before writing tests "
        "and only test what is actually implemented."
    ),
    llm=tester_llm,
    verbose=False,
)

fixer = Agent(
    role="Bug Fixer",
    goal=(
        "Read the original code, the review findings, and the test suite. "
        "Produce a corrected final version of the code that passes all tests "
        "and addresses all review comments."
    ),
    backstory=(
        "You are a senior developer who specialises in fixing bugs and hardening code. "
        "You receive code, a review, and a test suite, then return a clean final version "
        "that satisfies the reviewer and would pass the tests. "
        "You do not rewrite things that are already correct — only fix what is flagged."
    ),
    llm=coder_llm,
    verbose=False,
)

integrator = Agent(
    role="Integration Engineer",
    goal=(
        "Verify that the implementation and test suite are properly wired together. "
        "Fix any import errors, missing dependencies, or structural issues that would "
        "prevent the tests from running at all."
    ),
    backstory=(
        "You are an integration engineer whose only job is making sure code actually runs. "
        "You check that all imports are correct, that test files import from the right modules, "
        "that function/class names match between implementation and tests, "
        "and that there are no obvious runtime errors before pytest is even invoked. "
        "You output TWO files clearly separated: "
        "1. The final implementation.py "
        "2. The final tests.py with correct imports at the top."
    ),
    llm=coder_llm,
    verbose=False,
)
