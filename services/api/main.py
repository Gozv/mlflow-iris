from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Cargar el modelo desde MLflow (reemplaza <RUN_ID> con uno real)
model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")

@app.post("/predict")
async def predict(features: IrisFeatures):
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}

@app.get("/health")
async def health():
    return {"status": "ok"}