#from airflow.decorators import dag, task
from datetime import datetime
import pickle
import pandas as pd
import sys


#@dag(
    #dag_id="scoring",
    #description="Scoring DAG",
    #start_date=datetime(2025, 6, 15),
    #schedule_interval="@daily",
    #tags=["mlops", "scoring"],
#)

def pipeline():

    #@task
    def read_data(filename):
        df = pd.read_parquet(filename)
        categorical = ['PULocationID', 'DOLocationID']
        
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df

    

    #@task
    def make_prediction(df): 
        with open('model.bin', 'rb') as f_in:
            dv, model = pickle.load(f_in)
        categorical = ['PULocationID', 'DOLocationID']
        dicts = df[categorical].to_dict(orient='records')
        X_val = dv.transform(dicts)
        y_pred = model.predict(X_val) #The standard deviation of the prediction is: 6.24 (y_pred.std())
        print("The ride will take {} minutes".format(y_pred.mean()))
        print("The standard devaition of the prediction is: {}".format(y_pred.std())) 
        
    year = 2023
    month = 3

    df  = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{int(year):04d}-{int(month):02d}.parquet')
    make_prediction(df)

dag_instance = pipeline()

