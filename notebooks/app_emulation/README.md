This file is faking process oriented "app-like" behaviour to allow simulation of app flow across jupyter notebooks for later integration into real application layer.
All files here are maintained and require more strict rules in regards of updates and code changes. Please be more careful.

# Process Overview
[will receive regular updates, something we need to iterate and discuss]

1. Preprocessing (Prepare dataset for model training)
1.1. 'prep_database_file_generator' Creation of base file used as starting point of data input stream
1.2. 'prep_database_file_merger' Optional step - used to merge base files
1.3. 'prep_data_analysis' Preprocessing Data Analysis (optional, only visualizations and reports, no generated files for later usage)
1.4. 'prep_database_file_split' Preprocessing split into Combustion and Electric datasets
2. Model Training & Evaluation
3. tbd

# General Rules

## Notebooks and Python files
* Docstring with brief description of Purpose, Input, Output, Setup/Config (e.g. variables we want to adjust and inherit globally)
* Prefer smaller units and multiple files and try to avoid monolith

## Config
save shared variables (e.g. parameters for models, filepaths, etc.) in config.py
(later we can consider using .json file)

## Folders
Files such as datasets, images, etc. will not be tracked through git.
Still be aware and obey folder structure flow as we will work with relative paths when working with files.

### files
All files that are used somewhere as input (also output of certain process step can become input of a following process step)

### output
All files that are not used somewhere else as input (**only** output, e.g. final results)
