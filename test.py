from starter.starter.ml.model import (train_model,
                                      compute_model_metrics,
                                      load_model)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def test_train_model():
    # Generate some synthetic data for testing
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train the random forest classifier using the provided function
    model = train_model(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(accuracy)

    assert isinstance(
        model,
        RandomForestClassifier) and isinstance(
        accuracy,
        float)


def test_compute_model_metrics():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    # Calculate metrics using the provided function
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Define threshold values for passing the test (you can adjust these as
    # needed)
    precision_threshold = 0.8
    recall_threshold = 0.6
    fbeta_threshold = 0.8

    # Check if precision, recall, and F-beta score meet the thresholds
    assert (precision >= precision_threshold and recall >= recall_threshold and
            fbeta >= fbeta_threshold)


def test_load_model():
    model_path = 'starter/model/'
    model, _, _ = load_model(f"{model_path}")

    assert isinstance(model, RandomForestClassifier)


def test_load_encoder():
    model_path = 'starter/model/'
    _, encoder, _ = load_model(f"{model_path}")

    assert encoder is not None


def test_load_lb():
    model_path = 'starter/model/'
    _, _, lb = load_model(f"{model_path}")

    assert lb is not None
