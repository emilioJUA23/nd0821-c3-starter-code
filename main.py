# Put the code for your API here.
from fastapi import FastAPI, HTTPException
from typing import Union, Optional
# BaseModel from Pydantic is used to define data objects
from pydantic import BaseModel
import pandas as pd
import os, pickle
from .starter.ml.data import process_data
from .starter.ml.model import load_model, inference

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

class BaseInferenceObject(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

app = FastAPI()
model, encoder, lb = load_model('./starter/model/')



@app.get("/")
async def root():
    return 'Hello from the inferenci pipeline.'


@app.post("/inference/")
def infer(data: BaseInferenceObject):
    obj = data.dict()
    proccessed = pd.DataFrame([obj])
    proccessed = proccessed.rename(columns={
        "marital_status":"marital-status",
        "native_country":"native-country"
        })
    print(proccessed)
    proccessed,_,_,_ = process_data(proccessed, cat_features, training=False, encoder=encoder, lb=lb)
    preds = model.predict(proccessed)
    return f"{preds}"
