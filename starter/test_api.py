
"""
Script to test api
author: Luis Emilio Juarez
date: 2040-05-04
"""

import requests
import json

#url = "enter heroku web app url here"
url = "http://localhost:8000/inference/"

def test_approve():
    # explicit the sample to perform inference on
    sample =  {
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
    data = json.dumps(sample)
    
    # post to API and collect response
    response = requests.post(url, data=data )
    
    # display output - response will show sample details + model prediction added
    assert response.status_code == 200 and response.text == '"[1]"' 


def test_fail():
    # explicit the sample to perform inference on
    sample =  {
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
    data = json.dumps(sample)
    
    # post to API and collect response
    response = requests.post(url, data=data )
    
    # display output - response will show sample details + model prediction added
    assert response.status_code == 200 and response.text == '"[0]"' 