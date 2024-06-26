import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import logging
import joblib
from starter.starter.ml.data import process_data
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
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    logger.info("Best Hyperparameters:")
    logger.info(grid_search.best_params_)
    return best_model


def load_model(model_path):
    return joblib.load(f"{model_path}model.pkl"), joblib.load(
        f"{model_path}encoder.pkl"), joblib.load(f"{model_path}lb.pkl")


def save_model(model, encoder, lb, model_path):
    joblib.dump(model, f"{model_path}model.pkl")
    joblib.dump(encoder, f"{model_path}encoder.pkl")
    joblib.dump(lb, f"{model_path}lb.pkl")


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning
    model using precision, recall, and F1.

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


def compute_slices(df, feature, y, preds):
    """
    Compute the performance on slices for a given categorical feature
    a slice corresponds to one value option of the categorical feature analyzed
    """
    print(df)
    slice_options = list(set(df[feature]))
    perf_df = pd.DataFrame(
        index=slice_options,
        columns=['feature', 'n_samples',
                 'precision', 'recall', 'fbeta'])
    for option in slice_options:
        slice_mask = df[feature] == option

        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)

        perf_df.at[option, 'feature'] = feature
        perf_df.at[option, 'n_samples'] = len(slice_y)
        perf_df.at[option, 'precision'] = precision
        perf_df.at[option, 'recall'] = recall
        perf_df.at[option, 'fbeta'] = fbeta

    # reorder columns in performance dataframe
    perf_df.reset_index(names='feature value', inplace=True)
    colList = list(perf_df.columns)
    colList[0], colList[1] = colList[1], colList[0]
    perf_df = perf_df[colList]

    return perf_df


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

    X, y, encoder, lb = process_data(df, cat_feat, 'salary')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(y)
    print(X)
    model = train_model(X_train, y_train)
    preds = inference(model, X)
    metrics = compute_model_metrics(y, preds)
    save_model(model, encoder, lb, model_path)
    print(metrics)
    with open('slice_output.txt', 'w') as file:
        for feat in cat_feat:
            f_df = compute_slices(df, feat, y, preds)
            file.write(f"{feat}\n")
            file.write("\n")
            file.write(f_df.to_string())
            file.write("\n")
