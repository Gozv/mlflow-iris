import streamlit as st
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

st.title("🚀 Dashboard de Monitoreo de Modelos Iris")

# Conectar a MLflow
mlflow.set_tracking_uri("http://localhost:5000")
runs = mlflow.search_runs()

if not runs.empty:
    st.subheader("Últimas Ejecuciones")
    st.dataframe(runs[['run_id', 'metrics.accuracy', 'start_time']])
    
    st.subheader("Precisión Histórica")
    fig, ax = plt.subplots()
    ax.plot(runs['start_time'], runs['metrics.accuracy'], marker='o')
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("No hay ejecuciones registradas en MLflow.")