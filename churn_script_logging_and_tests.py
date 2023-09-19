"""
Test suite for customer churn helper library with logging.

Author: Felix Abrecht
12 May 2023
"""
import os
import logging
import joblib
import churn_library as cl


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    err = None
    try:
        perform_eda(df)
        assert os.path.isfile('./images/eda/churn_hist.png')
        assert os.path.isfile('./images/eda/age_hist.png')
        assert os.path.isfile('./images/eda/marital_status.png')
        assert os.path.isfile('./images/eda/total_trans_ct.png')
        assert os.path.isfile('./images/eda/correlation.png')
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_eda: One or more plots were not created.")
        raise err

    if err is None:
        logging.info("Testing perform_eda: SUCCESS")


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    '''
    err = None
    try:
        assert all([col in list(df.columns) for col in cl.CAT_COLS])
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: One or more columns are not categorical.")
        raise err

    try:
        df_encoded = encoder_helper(df, cl.CAT_COLS)
    except Exception as err:
        logging.error(
            "Testing encoder_helper: Error while trying to create df with new columns.")
        raise err

    try:
        assert all([col + '_Churn' in list(df_encoded.columns)
                   for col in cl.CAT_COLS])
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Output columns have wrong suffix or do not exists.")
        raise err

    if err is None:
        logging.info("Testing encoder_helper: SUCCESS")

def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    err = None
    try:
        assert(cl.perform_feature_engineering(df, cl.KEEP_COLS))
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: FAIL. Check if train_test_split is called correctly.')

    if err is None:
        logging.info("Testing perform_feature_engineering: SUCCESS")


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    err = None

    try:
        cl.train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile('images/results/roc_curve_lrc.png')
        assert os.path.isfile('images/results/roc_curves_lrc_rfc.png')
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile('images/results/shap_summary_plot.png')
        assert os.path.isfile('images/results/classification_report.png')
    except FileNotFoundError as err:
        logging.error(
            "Testing train_models: Result plots and/or pkl files have not been created.")
        raise err

    if err is None:
        logging.info("Testing train_models: SUCCESS")


def test_feature_importance_plot(feature_importance_plot, rfc_model, X_test):
    '''
    test feature_importance_plot
    '''
    err = None

    try:
        feature_importance_plot(
            rfc_model,
            X_test,
            'images/results/feature_importance_rfc.png')
        assert os.path.isfile('images/results/feature_importance_rfc.png')
    except FileNotFoundError as err:
        logging.error(
            "Testing feature_importance_plot: Result plot not created.")
        raise err
    if err is None:
        logging.info("Testing feature_importance_plot: SUCCESS")


if __name__ == "__main__":
    test_import(cl.import_data)

    df = cl.import_data(cl.DATA_PATH)
    test_eda(cl.perform_eda, df)

    test_encoder_helper(cl.encoder_helper, df)
    df_encode = cl.encoder_helper(df, cl.CAT_COLS)

    test_perform_feature_engineering(cl.perform_feature_engineering, df_encode)
    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
        df_encode, cl.KEEP_COLS)

    test_train_models(cl.train_models, X_train, X_test, y_train, y_test)

    rfc_model = joblib.load('./models/rfc_model.pkl')
    test_feature_importance_plot(cl.feature_importance_plot, rfc_model, X_test)
