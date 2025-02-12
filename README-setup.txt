1. Create venv folder
python -m venv venv

2. Activate venv (make sure to give execution permissions depending on os)
- usually VS Code should recognize this allowing you to use new venv as standard Python environment
- make sure you have active environment also in your terminal if you install packages or execute scripts
You can manually activate venv environment with: .\venv\Scripts\activate

3. Install requirement files (will be updated)
pip install -r requirements_base.txt
pip install -r requirements_development.txt

--> Future Linting and Formatting will be added.

General Rules for our Git Flow

create your own branch for now we use dev to signal we're on a development branch
git checkout -b dev-<your_name>
e.g.
git checkout -b dev-analyse2023

TODO:
- fetch
- merge
- pull ...