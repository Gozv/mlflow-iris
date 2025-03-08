#!/bin/bash

# Iniciar MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 &

# Entrenar el modelo
python flows/training_flow.py run

# Iniciar la API
uvicorn services.api.main:app --reload &

# Iniciar el dashboard
streamlit run dashboards/monitor.py