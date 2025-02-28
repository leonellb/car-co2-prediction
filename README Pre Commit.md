# DataScience

## TODO's
- update to Python version 3.11.9 - April 2, 2024

## CI/CD Setup
pre-commit
- https://pre-commit.com/#install
    pre-commit run --all-files

Ruff
- setup for Github actions: https://docs.astral.sh/ruff/integrations/#github-actions
- setup for pre-commit https://docs.astral.sh/ruff/integrations/#pre-commit
- settings (pyproject.toml): https://docs.astral.sh/ruff/settings/
- manual execution through terminal:
    ruff check --fix

Line endings
- Make sure to use proper line endings (recommended for cross-platform projects).
    git config --global core.autocrlf input

## Interesting ressources
- https://testdriven.io/blog/clean-code-python/
- https://testdriven.io/blog/python-code-quality/
