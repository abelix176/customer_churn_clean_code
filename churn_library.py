"""
Helper library to discern the likelyhood of customer churn.

Author: Felix Abrecht
12 May 2023
"""
from io import StringIO
import sys
import os
import argparse
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    adds 'Churn' to dataframe

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    eda_img = 'images/eda/'
    # plot and save churn hist
    fig = plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    fig.savefig(os.path.join(eda_img, 'churn_hist.png'))
    plt.close(fig)
    # plot and save age hist
    fig = plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    fig.savefig(os.path.join(eda_img, 'age_hist.png'))
    plt.close(fig)
    # plot and save marital status
    fig = plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    fig.savefig(os.path.join(eda_img, 'marital_status.png'))
    plt.close(fig)
    # plot and save total trans ct
    fig = plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    fig.savefig(os.path.join(eda_img, 'total_trans_ct.png'))
    plt.close(fig)
    # plot and save correlation
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig.savefig(os.path.join(eda_img, 'correlation.png'))
    plt.close(fig)


def encoder_helper(df, category_lst, response="_Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column], default is _Churn as suffix

    output:
            df: pandas dataframe with new columns for
    '''
    result = df.copy()
    for column in category_lst:
        new_column = column + response
        result[new_column] = result.groupby(column)["Churn"].transform("mean")
    return result


def perform_feature_engineering(df, keep_cols, test_size=0.3):
    '''
    input:
              df: pandas dataframe
              keep_cols: selected features for training
              test_size: used in train_test_split, default = 0.3

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df['Churn']
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()

    # Your print statements
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    # Restore stdout
    sys.stdout = old_stdout

    # Save the captured text as a PNG image
    output_text = buffer.getvalue()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.text(
        0,
        0.5,
        output_text,
        fontsize=12,
        family='monospace',
        ha='left',
        va='center',
        wrap=True)
    plt.savefig(
        'images/results/classification_report.png',
        bbox_inches='tight',
        dpi=300)
    plt.close(fig)


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Plot results
    # ROC plot for logistic regression only
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.savefig('images/results/roc_curve_lrc.png')

    # ROC plot for both lrc and random forest
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('images/results/roc_curves_lrc_rfc.png')
    plt.close()

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Explain Model - save the SHAP summary plot as a PNG file
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(
        'images/results/shap_summary_plot.png',
        bbox_inches='tight',
        dpi=300)
    plt.close()


# Code to execute the library and generate the results
DATA_PATH = 'data/bank_data.csv'

CAT_COLS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to read csv file from.")
    args = parser.parse_args()
    data_pth = args.data_path or DATA_PATH
    df = import_data(data_pth)
    perform_eda(df)

    df_encode = encoder_helper(df, CAT_COLS)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_encode, KEEP_COLS)

    train_models(X_train, X_test, y_train, y_test)

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    feature_importance_plot(
        rfc_model,
        X_test,
        'images/results/feature_importance_rfc.png')
