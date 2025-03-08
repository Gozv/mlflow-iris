from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel
from mlflow.tracking import MlflowClient

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Cargar el modelo más reciente de MLflow
client = MlflowClient()
latest_run = client.search_runs(
    experiment_ids=["0"],  # ID del experimento (por defecto "0" si no has creado otros)
    order_by=["attributes.start_time DESC"],
    max_results=1
)[0]

model = mlflow.sklearn.load_model(f"runs:/{latest_run.info.run_id}/model")

@app.post("/predict")
async def predict(features: IrisFeatures):
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}

@app.get("/health")
async def health():
    return {"status": "ok"}