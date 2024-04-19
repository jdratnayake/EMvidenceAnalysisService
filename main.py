from fastapi import FastAPI, Header, HTTPException
import sys
import os
import json
import numpy as np
from tensorflow import keras
from collections import Counter
from fastapi.responses import StreamingResponse
import gzip
import bz2
import time
import os

# Create an instance of FastAPI
app = FastAPI()

# Define a route for root endpoint
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.get("/anomaly")
async def predict_anomaly():
    em_preprocessing_file_path = "./em_preprocessed/class_8_iphone4s_sms-app.npy"
    analysis_plugin_ml_model_path = "./ml_models/apple_iphone_4s__detect_anomalies__neural_network_model.h5"

    # 0 = Normal behavior
    # 1 = Anomaly
    labels = ["Original Firmware", "Anomaly"]


    X = np.load(em_preprocessing_file_path)
    model = keras.models.load_model(analysis_plugin_ml_model_path)
    y = model.predict(X)

    # Select the index of the highest probability class for each sample in y
    y_pred = np.argmax(y, axis=1)
    classes_counts = Counter(y_pred) # Count occurrences of elements in y_pred
    classes_counts_dict = {int(key): int(value) for key, value in classes_counts.items()}
    sum_of_counts = sum(classes_counts_dict.values())
    classes_counts_dict_sorted = {labels[k]: round(((classes_counts_dict[k] * 100.0) / sum_of_counts), 2) for k, v in sorted(classes_counts_dict.items(), key=lambda item: item[1], reverse=True)}

    output = json.dumps(classes_counts_dict_sorted)
    return output

@app.get("/behaviour")
async def predict_anomaly():
    em_preprocessing_file_path = "./em_preprocessed/class_8_iphone4s_sms-app.npy"
    analysis_plugin_ml_model_path = "./ml_models/apple_iphone_4s__detect_behaviour_of_6_classes__neural_network_model.h5"

    labels = ["Using Calendar App", "Using Email App", "In Home Screen", "Using SMS App", "Using Gallary App", "Idle"]

    X = np.load(em_preprocessing_file_path)
    model = keras.models.load_model(analysis_plugin_ml_model_path)
    y = model.predict(X)

    # Select the index of the highest probability class for each sample in y
    y_pred = np.argmax(y, axis=1)
    classes_counts = Counter(y_pred) # Count occurrences of elements in y_pred
    classes_counts_dict = {int(key): int(value) for key, value in classes_counts.items()}
    sum_of_counts = sum(classes_counts_dict.values())
    classes_counts_dict_sorted = {labels[k]: round(((classes_counts_dict[k] * 100.0) / sum_of_counts), 2) for k, v in sorted(classes_counts_dict.items(), key=lambda item: item[1], reverse=True)}

    output = json.dumps(classes_counts_dict_sorted)
    return output

@app.get("/compression")
async def predict_anomaly(compression: str = Header(None), filename: str = Header(None)):
    file_path = "em_raw/" + filename
    output_file = ""

    start_time = time.time()

    
    if compression == "gzip":
        output_file = "output.gz"
        with open(file_path, 'rb') as f_in:
            with gzip.open(output_file, 'wb') as f_out:
                f_out.writelines(f_in)
    elif compression == "bzip2":
        output_file = "output.bz2"
        with open(file_path, 'rb') as f_in:
            with bz2.open(output_file, 'wb') as f_out:
                f_out.writelines(f_in)
    else:
        raise HTTPException(status_code=400, detail="Invalid compression algorithm. Please use 'gzip' or 'bzip2'.")

    end_time = time.time()
    time_taken = end_time - start_time
    compressed_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    return {
        "time_taken": time_taken,
        "compressed_size": compressed_size_mb,
    }