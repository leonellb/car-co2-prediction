# DataScience

## TODO's
- investigate update to Python version 3.11.9 - April 2, 2024

## Repos Initial Setup

1. Create venv folder
python -m venv venv

2. Activate venv (make sure to give execution permissions depending on os)
- usually VS Code should recognize this allowing you to use new venv as standard Python environment
- make sure you have active environment also in your terminal if you install packages or execute scripts
You can manually activate venv environment with: .\venv\Scripts\activate

3. Install requirement files (will be updated)
pip install -r requirements_base.txt
pip install -r requirements_development.txt

4. Install to activate pre-commit hooks
pre-commit install

--> Future Linting and Formatting will be added through pre-commit config file (.pre-commit-config.yaml) and related package config files such as pyproject.toml.

## CI/CD Setup

### pre-commit
- https://pre-commit.com/#install to execute pre-commit hook locally run:
    pre-commit run --all-files

### Ruff
- setup for Github actions: https://docs.astral.sh/ruff/integrations/#github-actions
- setup for pre-commit https://docs.astral.sh/ruff/integrations/#pre-commit
- settings (pyproject.toml): https://docs.astral.sh/ruff/settings/
- manual execution through terminal:
    ruff check --fix

### Line endings
- Make sure to use proper line endings (recommended for cross-platform projects).
    git config --global core.autocrlf input

## General Rules for our Git Flow

Just basic principles, we did not agree on a a real Gitflow to keep things simple:

* create your own branch for now we use "dev-" to signal we're on a development branch
git checkout -b dev-<your_name>
e.g.
git checkout -b dev-analyse2023

* once finished (and clean) merge your code into main:
git checkout main
git merge <your_branch_name>

* to merge code from main into your branch
git merge origin <from_branch_name>
git merge origin main

* only merge relevant content and keep anything else local (e.g. custom csv files or python experiements for individual usecases/experiments)

## Interesting ressources
- https://testdriven.io/blog/clean-code-python/
- https://testdriven.io/blog/python-code-quality/
