set windows-shell := ["pwsh", "-NoProfile", "-Command"]

default: verify

# Display the current repository state
status:
    git status --short --branch

# Apply Ruff's formatter to Python files
format:
    ruff format .

# Run Ruff's linter
lint:
    ruff check .

# Verify that Ruff formatting is already clean without changing files
format-check:
    ruff format --check .

# Run the non-mutating code-quality gates
verify: lint format-check
