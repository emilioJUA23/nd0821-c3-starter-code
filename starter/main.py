# Put the code for your API here.
from fastapi import FastAPI, HTTPException
from typing import Union, Optional
# BaseModel from Pydantic is used to define data objects
from pydantic import BaseModel
import pandas as pd
import os, pickle
from starter.ml.data import process_data


class InputData(BaseModel):
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


@app.get("/")
async def root():
    return {"message": "Hello World"}
