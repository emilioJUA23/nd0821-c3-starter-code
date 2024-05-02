import pandas as pd
from scipy.sparse import data
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import logging
import joblib
from starter.ml.data import process_data
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    logger.info("Training model with hyperparameter tunning...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    logger.info("Best Hyperparameters:")
    logger.info(grid_search.best_params_)
    return best_model


def load_model(model_path):
    return joblib.load(model_path)


def save_model(model, model_path):
    joblib.dump(model, model_path)



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_metrics_cat_features(model, X, y, cat_features):
    logger.info("Getting overall metrics...")
    metrics = {}
    preds = inference(model, X)
    overall_precision, overall_recall, overall_fbeta = compute_model_metrics(y, preds)
    metrics['overall'] = {'precision': overall_precision, 'recall': overall_recall, 'fbeta': overall_fbeta}

    logger.info("Calculating metrics for each slice...")
    for cat_feature in cat_features:
        columns = X.filter(like=cat_feature).columns
        for column in columns:
            mask = (X[column] == 1)
            X_slice = X[mask]
            y_slice = y[mask]
            if not X_slice.empty:
                preds_slice = inference(model, X_slice)
                precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
                metrics[column] = {'precision': precision, 'recall': recall, 'fbeta': fbeta}

    return metrics



def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


if __name__ == "__main__":
    data_path = '../../data/census.csv'
    model_path = '../../model/'
    df = pd.read_csv(data_path)
    cat_feat = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X,y,encoder,lb = process_data(df, cat_feat, 'salary')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(y)
    print(X)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    metrics = compute_model_metrics(y_test, preds)
    save_model(model, f"{model_path}model.pkl")
    print(metrics)
