# Predicting Car CO₂ Emissions through Machine Learning

This project explores data-driven methods to estimate CO₂ emissions from newly registered vehicles in the European Union using machine learning and deep learning techniques. Relying on structured datasets from the European Environment Agency (EEA), it aims to build accurate predictive models for both electric and combustion vehicles. The study compares a variety of algorithms — from traditional regressors to neural networks — to predict electric energy consumption and specific CO₂ emissions, while also classifying fuel types. In doing so, it provides valuable insights into the relationship between vehicle specifications and environmental impact, supporting manufacturers and policymakers in aligning with sustainable mobility goals.

This project was developed as part of the [DataScientest](https://www.datascientest.com/) Data Science Bootcamp, in partnership with Université Paris 1 Panthéon-Sorbonne. Future improvements are planned, including restructuring the final report and building an interactive Streamlit-based web application to showcase the results dynamically.

---

## Project Overview

The repository is structured into three core modeling tracks:

1. **[Electric Energy Consumption (main project)](https://github.com/DataScientest-Studio/JAN25_BDS_INT_CO2/tree/main/notebooks/app_emulation)**  
   Initial preprocessing and modeling pipeline for predicting electric energy consumption. Also serves as the foundation for the future Streamlit application.

2. **[Alternative Electric Models](https://github.com/DataScientest-Studio/JAN25_BDS_INT_CO2/tree/main/notebooks/electric_energy_consumption_project_2)**  
   Uses a modified dataset that retains the `fuel_consumption` variable to explore additional preprocessing techniques and model types.

3. **[CO₂ Emissions Modeling](https://github.com/DataScientest-Studio/JAN25_BDS_INT_CO2/tree/main/notebooks/)**  
   Contains models for predicting CO₂ emissions in combustion vehicles using merged and preprocessed datasets.

Reports and the final presentation are available in the [`reports/`](https://github.com/DataScientest-Studio/JAN25_BDS_INT_CO2/tree/main/reports) folder.  
A Streamlit-based interface for interactive exploration will be developed and integrated soon.

---

## Project Organization

```text
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data               <- Should be in your computer but not on GitHub (only in .gitignore)
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's name, and a short `-` delimited description.
│
├── references         <- Data dictionaries, manuals, links, and other explanatory materials.
│
├── reports            <- Project reports and presentations
│   └── figures        <- Generated graphics and figures used in reporting
│
├── requirements.txt   <- Environment dependencies (`pip freeze > requirements.txt`)
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── features       <- Scripts to turn raw data into features for modeling
│   ├── models         <- Scripts to train and run models
│   ├── visualization  <- Scripts for exploratory and result-oriented visualizations

    │   │   └── visualize.py
```

---

## How to Run Locally

notebooks/app_emulation/README.md

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
