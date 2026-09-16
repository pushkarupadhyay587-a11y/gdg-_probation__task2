from fastapi import FastAPI 
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import JSONResponse
from typing import Literal
from pathlib import Path
import joblib
import pandas as pd 


base_dir = Path(__file__).resolve().parent
model_path = base_dir / "models" / "prediction_model" / "svc.pkl"
preprocessor_path = base_dir / "models" / "preprocessor" / "preprocessor.pkl"
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)


class EVInput(BaseModel):
    Age: int = Field(..., gt=0, lt=120, description="Age of the person")
    Gender: Literal["Male", "Female", "Other"] = Field(..., description="Gender")
    Annual_Income_USD: float = Field(..., gt=0, description="Annual income in USD")
    City_Type: Literal["Urban", "Rural", "Suburban", "Semi-Urban"] = Field(..., description="Type of city")
    Daily_Commute_km: float = Field(..., ge=0, description="Daily commute distance in km")
    Number_of_Cars_Owned: int = Field(..., ge=0, description="Number of cars owned")
    Current_Car_Type: Literal["Truck", "SUV", "Sedan", "Hatchback"] = Field(..., description="Type of current car")
    Charging_Stations_Near_Home: int = Field(..., ge=0, description="Number of charging stations near home")
    Charging_Stations_Near_Work: int = Field(..., ge=0, description="Number of charging stations near work")
    Home_Charging_Possible: Literal["Yes", "No"] = Field(..., description="Is home charging possible?")
    Environmental_Concern_Level: int = Field(..., ge=1, le=10, description="Environmental concern level (1–10)")
    Subsidy_Available: Literal["Yes", "No"] = Field(..., description="Is subsidy available?")
    Range_Anxiety_Level: Literal["Low", "Medium", "High"] = Field(..., description="Range anxiety level of the customer")

    @field_validator("City_Type", mode="before")
    @classmethod
    def normalize_city_type(cls, value):
        if value == "Semi-Urban":
            return "Suburban"
        return value


app = FastAPI()

@app.get('/')
def home():
    return "hello! API is running"

@app.post('/predict')
def predict_ev(data: EVInput):
    input_df = pd.DataFrame([data.model_dump()])
    input_df["City_Type"] = input_df["City_Type"].replace({"Semi-Urban": "Suburban"})

    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)[0]

    return JSONResponse(status_code=200, content={'prediction': int(prediction)})
