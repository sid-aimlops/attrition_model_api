from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

# Define the input data model
class PredictionInput(BaseModel):
    feature: float

# Root URL route
@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}

# Prediction route
@app.post("/predict")
def predict(input_data: PredictionInput) -> Dict[str, str]:
    # Example of how you'd make a prediction using your model
    # Replace this with your actual prediction code
    prediction = "Yes" if input_data.feature > 0.5 else "No"
    
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)