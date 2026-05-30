from crewai import Agent, LLM

# ── LLMs ──────────────────────────────────────────────────────────────────────
# Using 9b for all agents so everything fits in VRAM and runs fast.
# Swap coder_llm to "ollama/qwen3-coder:30b" for higher quality (but slower).

BASE_MODEL = "ollama/qwen3-coder:30b"  # single model for all agents — no swap overhead

fast_llm = LLM(
    model=BASE_MODEL,                  # preferred: ollama/qwen3.5:9b
    base_url="http://localhost:11434",
    timeout=1200,                      # 20 min cap — avoids silent hangs
)

coder_llm = LLM(
    model=BASE_MODEL,                  # preferred: ollama/qwen3-coder:30b
    base_url="http://localhost:11434",
    timeout=1200,
)

tester_llm = LLM(
    model=BASE_MODEL,                  # preferred: ollama/qwen2.5-coder:7b
    base_url="http://localhost:11434",
    timeout=1200,
)

# ── Agents ────────────────────────────────────────────────────────────────────

coder = Agent(
    role="Senior Python Developer",
    goal=(
        "Write clean, working, well-documented Python code that solves the task exactly. "
        "Output ONLY the labelled file blocks — no explanations, no commentary, no prose."
    ),
    backstory=(
        "/no_think\n"
        "You are an experienced Python developer who writes production-quality code. "
        "You always include docstrings, handle edge cases, and follow PEP8. "
        "When you receive review feedback, you incorporate it fully. "
        "IMPORTANT: Your entire response must consist of labelled file blocks only. "
        "Do not write any text before, between, or after the file blocks."
    ),
    llm=coder_llm,
    verbose=False,
    respect_context_window=True,
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
    respect_context_window=True,
)

tester = Agent(
    role="QA Engineer",
    goal=(
        "Write comprehensive pytest tests that cover happy paths, edge cases, and error conditions, maximum 10 tests. "
        "Output ONLY the labelled test file block — no explanations, no commentary, no prose."
    ),
    backstory=(
        "/no_think\n"
        "You are a QA engineer who writes thorough test suites. "
        "You always use pytest fixtures, parametrize where appropriate, "
        "and make sure tests are isolated and repeatable. "
        "You read the implementation carefully before writing tests "
        "and only test what is actually implemented. "
        "IMPORTANT: Your entire response must consist of the labelled test file block only. "
        "Do not write any text before, between, or after the file block."
    ),
    llm=tester_llm,
    verbose=False,
    respect_context_window=True,
)

fixer = Agent(
    role="Bug Fixer",
    goal=(
        "Read the original code, the review findings, and the test suite. "
        "Produce a corrected final version of the code that passes all tests "
        "and addresses all review comments. "
        "Output ONLY the labelled file blocks — no explanations, no commentary, no prose."
    ),
    backstory=(
        "/no_think\n"
        "You are a senior developer who specialises in fixing bugs and hardening code. "
        "You receive code, a review, and a test suite, then return a clean final version "
        "that satisfies the reviewer and would pass the tests. "
        "You do not rewrite things that are already correct — only fix what is flagged. "
        "IMPORTANT: Your entire response must consist of labelled file blocks only. "
        "Do not write any text before, between, or after the file blocks."
    ),
    llm=coder_llm,
    verbose=False,
    respect_context_window=True,
)

integrator = Agent(
    role="Integration Engineer",
    goal=(
        "Verify that the implementation and test suite are properly wired together. "
        "Fix any import errors, missing dependencies, or structural issues that would "
        "prevent the tests from running at all. "
        "Output ONLY the labelled file blocks — no explanations, no commentary, no prose."
    ),
    backstory=(
        "/no_think\n"
        "You are an integration engineer whose only job is making sure code actually runs. "
        "You check that all imports are correct, that test files import from the right modules, "
        "that function/class names match between implementation and tests, "
        "and that there are no obvious runtime errors before pytest is even invoked. "
        "IMPORTANT: Your entire response must consist of labelled file blocks only. "
        "Do not write any text before, between, or after the file blocks."
    ),
    llm=coder_llm,
    verbose=False,
    respect_context_window=True,
)
