from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

class IrisTrainingFlow(FlowSpec):
    data_path = Parameter('data', default='data/iris.csv')
    
    @step
    def start(self):
        self.df = pd.read_csv(self.data_path)
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        X = self.df.drop('species', axis=1)
        y = self.df['species'].astype('category').cat.codes
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)
        self.next(self.train_model)
    
    @step
    def train_model(self):
        mlflow.set_tracking_uri("http://localhost:5000")  # Asegúrate de tener MLflow corriendo
        mlflow.set_experiment("Iris-Classification")
        
        with mlflow.start_run():
            self.model = RandomForestClassifier(n_estimators=100)
            self.model.fit(self.X_train, self.y_train)
            
            # Logging en MLflow
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(self.model, "model")
        
        self.next(self.end)
    
    @step
    def end(self):
        print("Entrenamiento completado y modelo registrado en MLflow!")

if __name__ == '__main__':
    IrisTrainingFlow()