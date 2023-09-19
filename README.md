# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This Python library is aimed at performing exploratory data analysis and building predictive models on banking data. The purpose of the project is to predict customer churn, an important metric for any business, especially in the banking sector. The project involves data import, data exploration, feature engineering, model training (including logistic regression and random forest), and model evaluation.

## Files and data description
This project is organized as follows:
- The Python script churn_library.py is intended to be run from the command line (see below) or used as a helper lib for companies who want to find customers who are likely to churn.
- The script reads data from the specified CSV file.
- Exploratory data analysis is performed and figures are saved in the images/eda/ directory.
- Categorical features are encoded and feature engineering is performed.
- Two predictive models (Random Forest and Logistic Regression) are trained.
- The models' performance metrics are evaluated and saved as images in the images/results/ directory.
- The trained models are saved in the models/ directory.
- SHAP values are computed and a summary plot is saved in the images/results/ directory.

The script content was refactored from churn_notebook.ipynb



## Running Files
The necessary dependencies can be installed using the respective 'requirements_.txt' file for the specified python version. The project was tested with version 3.6:
```python -m pip install -r requirements_py3.6.txt```

The above workflow can be executed via:
```ipython churn_library.py``` or ```python churn_library.py``` depending on your preferred/installed kernel.

Tests can be executed in a similar fashion:
```ipython chrun_script_logging_and_tests.py```. 
The logs can be found in the '/logs' folder.

## Pep8 and Pylint
Files are formatted following the pep8 guidline using autopep8. Both files, churn_library.py and churn_script_logging_and_tests.py, achieve a pylint scoring above 7.



