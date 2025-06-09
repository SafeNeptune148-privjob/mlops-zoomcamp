# Homework 3: Duration prediction model pipeline
# Author: Abraham Alvarado Padilla
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.exceptions import AirflowFailException
from datetime import datetime
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow 
from mlflow.tracking import MlflowClient
import os


@dag(
    dag_id="Homework_3",
    description="Homework 3: Duration prediction model pipeline",
    start_date=datetime(2025, 06, 07),
    schedule_interval="@daily",
    tags=["mlops","homework_3"],
    catchup=False
)


def pipeline():
    # Set up MLflow experiment and tracking
    mlflow.set_experiment("Homework_3_First_Pipeline")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.sklearn.autolog()
    os.makedirs("artifacts", exist_ok=True)

    # Answer to Question 3: 3,403,766

    # Code we put in the pipeline
    @task
    def read_dataframe():
        try:
            data_path = Variable.get("data_path")
        except KeyError:
            raise AirflowFailException("Airflow variable 'data_path' is not set.")
        if not os.path.exists(data_path):
            raise AirflowFailException(f"File not found: {data_path}")
        df = pd.read_parquet(data_path)
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        categorical = ['PULocationID', 'DOLocationID']
        numerical = ["trip_distance"]
        df[categorical] = df[categorical].astype(str)
        return {
            "features": df[categorical + numerical].to_dict(orient="records"),
            "target": df["duration"].tolist()
        }

    # Answer to Question 4: 3,316,216

    @task
    def train_model(df_train):
        with mlflow.start_run():
            train_dicts = pd.DataFrame(df_train["features"])
            dv = DictVectorizer()
            X_train = dv.fit_transform(train_dicts)
            y_train = pd.Series(df_train["target"])
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            with open("artifacts/dictvectorizer.pkl", "wb") as f_out:
                pk.dump(dv, f_out)
            mlflow.log_artifact("artifacts/dictvectorizer.pkl", artifact_path="preprocessors")
            
        client = MlflowClient()
        experiment = client.get_experiment_by_name("Homework_3_First_Pipeline")
        model_id = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.training_root_mean_squared_error > 0",
            order_by=["metrics.training_root_mean_squared_error ASC"],
            max_results=1
        )
        return model_id[0].info.run_id

    # The interception I got is 23.8483

    # Let's register the model
    @task
    def register_model(model_id_link):
        
        mlflow.register_model(
            model_uri=f"runs:/{model_id_link}/model",
            name="Best_Linearregression_prediction_model"
        ) 
    # Task Flow  
    data = read_dataframe()
    model_id_link = train_model(data)
    register_model(model_id_link)
            

dag_instance = pipeline()
