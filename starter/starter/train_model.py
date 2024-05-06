# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from starter.ml.model import train_model, save_model
import pandas as pd
from starter.ml.data import process_data
# Add the necessary imports for the starter code.

# Add code to load in the data.
data_path = '../../data/census.csv'
model_path = '../../model/'
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.

model = train_model(X_train, y_train)

model_path = '../model/'
save_model(model, f"{model_path}model.pkl")
