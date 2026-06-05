import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

class CustomerData(BaseModel):
    tenure: int = Field(..., example=12)
    MonthlyCharges: float = Field(..., example=70.5)
    TotalCharges: float = Field(..., example=846.0)
    Contract: str = Field(..., example="Month-to-month")
    TechSupport: str = Field(..., example="No")

app = FastAPI(title="Customer Churn Prediction Service")
model = joblib.load("models/churn_pipeline.pkl")

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        input_data = pd.DataFrame([customer.model_dump()])
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])
        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "actionable_risk": "High Risk" if probability >= 0.6 else "Low/Medium Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)