#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify




def read_data(filename):
    df = pd.read_parquet(filename)
    categorical = ['PUlocationID', 'DOlocationID']
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def prediction(yr, mo):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    # print('File opened')
    categorical = ['PUlocationID', 'DOlocationID']
    date = datetime(year=yr, month=mo, day=1)
    year = date.year
    month = date.month
    # print(f"{month:02d}")
    # print(month)
    df = read_data(f'data/fhv_tripdata_{year}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    # print(f"The mean predicted duration for {year:04d}/{month:02d} is {np.mean(y_pred)}")

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = df[['ride_id']].copy()


    df_result['predictions'] = y_pred
    # print('prediction done')
    return np.mean(y_pred)

"""
df_result.to_parquet(
    f'df_result_{year:04d}_{month:02d}.parquet',
    engine='pyarrow',
    compression=None,
    index=False)
"""


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    
    pred = prediction(ride['year'], ride['month'])

    result = {
        'mean prediction': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)