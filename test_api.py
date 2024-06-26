
"""
Script to test api
author: Luis Emilio Juarez
date: 2040-05-04
"""

from fastapi.testclient import TestClient
from main import app
client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_approve():
    # explicit the sample to perform inference on
    sample = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States"

    }
    # post to API and collect response
    response = client.post('/inference/', json=sample)

    # display output - response will show sample details + model prediction
    # added
    assert response.status_code == 200 and response.text == '"[1]"'


def test_fail():
    # explicit the sample to perform inference on
    sample = {
        "age": 20,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 20,
        "native_country": "United-States"

    }
    # post to API and collect response
    response = client.post('/inference/', json=sample)

    # display output - response will show sample details + model prediction
    # added
    assert response.status_code == 200 and response.text == '"[0]"'
