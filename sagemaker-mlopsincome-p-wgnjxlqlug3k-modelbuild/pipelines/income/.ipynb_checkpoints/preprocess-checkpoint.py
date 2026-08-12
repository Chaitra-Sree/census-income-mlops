

"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile
import pickle
import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    print(input_data)
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/raw-data.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")


    # Read the downloaded CSV file
    data = pd.read_csv(fn)
    

    # drop the "Phone" feature column
    data.replace("?", np.nan, inplace=True)
    for x in data.columns[data.isnull().sum() !=0]:
        data[x].fillna(data[x].mode()[0], inplace=True)
        
    categorical = data.columns[data.dtypes==object]
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    for c in data.columns[data.dtypes==object]:
      encoder.fit(data[c])
      data[c] = encoder.transform(data[c])
      mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
      print(c,":",mapping)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data[["age", "hours.per.week","fnlwgt","capital.loss"]] = scaler.fit_transform(data[["age", "hours.per.week","fnlwgt","capital.loss"]])
    
    zero = data[data.income==0]
    one = data[data.income==1]
    
    from sklearn.utils import resample
    oversample = resample(one, replace=True, n_samples=len(zero))
    data2 = pd.concat([oversample, zero], axis=0)

    x = data2.drop(["income"],axis=1)
    y = data2.income

    logger.info("going into train test split")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.4)


    trans = {
        'One_Hot': encoder,
        'scaler': scaler,
    }   

    # Split the data
    pd.DataFrame(np.c_[y_train, X_train]).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(np.c_[y_val, X_val]).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(np.c_[y_test, X_test]).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

