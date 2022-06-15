from asyncio.log import logger
import pandas as pd
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import get_run_logger 
from datetime import timedelta
import datetime as dt
import sys


import pickle
import os

# import mlflow
# mlflow.set_tracking_uri('http://127.0.0.1:5000')
# mlflow.set_experiment("nyc-taxi-regression-assignment-week3-q2")

# logger = get_run_logger()
# logger.info(f"mse = {mse}")
@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def get_paths(date=None):
    if date is None:
        today = dt.date.today()
        two_months_back = (today - timedelta(days=60)).strftime("%Y-%m")
        one_months_back = (today - timedelta(days=30)).strftime("%Y-%m")
        train_path = "./data/fhv_tripdata_{}.parquet".format(two_months_back)
        valid_path = "./data/fhv_tripdata_{}.parquet".format(one_months_back)
        return train_path, valid_path
    else:
        #convert date string to datetime object
        date = dt.datetime.strptime(date, "%Y-%m-%d")
        print(date)
        two_months_back = (date - timedelta(days=60)).strftime("%Y-%m")
        one_months_back = (date - timedelta(days=30)).strftime("%Y-%m")
        train_path = "./data/fhv_tripdata_{}.parquet".format(two_months_back)
        valid_path = "./data/fhv_tripdata_{}.parquet".format(one_months_back)
        return train_path, valid_path

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    logger = get_run_logger()
    

    mean_duration = df.duration.mean()
    if train:
        #change print statement to prefect logger

        #print(f"The mean duration of training is {mean_duration}")
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        # print(f"The mean duration of validation is {mean_duration}")
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    # with mlflow.start_run():
        # mlflow.set_tag("model", "linreg")
        
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values
    logger = get_run_logger()
    # print(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The shape of X_train is {X_train.shape}")
    #print(f"The DictVectorizer has {len(dv.feature_names_)} features")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)

    rmse = mean_squared_error(y_train, y_pred, squared=False)
    #print(f"The RMSE of training is: {rmse}")
    logger.info(f"The RMSE of training is: {rmse}")
    logger.info(f"Size of Dict Vectorizer {sys.getsizeof(dv)}")
    # mlflow.log_metric("training_rmse", rmse)
    # with open("models/preprocessor.b", "wb") as f_out:
    #     pickle.dump(dv, f_out)
    # mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values
    logger = get_run_logger()
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    # print(f"The RMSE of validation is: {rmse}")
    logger.info(f"The RMSE of validation is: {rmse}")
    # print(rmse)
    return
@flow
def main(date="2021-08-15"):
#    with mlflow.start_run():
        # mlflow.log_params({"date": date})
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    
    with open(f"models/model-"+str(date)+".bin", "wb") as f_out:
        pickle.dump(lr, f_out)
    with open(f"models/dv-"+str(date)+".b", "wb") as f_out:
        pickle.dump(dv, f_out)

# main()
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule, CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron="0 9 15 * * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)

