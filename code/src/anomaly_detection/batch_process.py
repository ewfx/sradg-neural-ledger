import os
import pandas as pd
from .anomaly_detector import detect_anomalies
import json

data_directory = "data/historical_data/"

def batch_anomaly_detection(case_name):
    with open("config/conf.json", "r") as f:
        config = json.load(f)
    
    for file in os.listdir(data_directory):
        if file.endswith(".csv") or file.endswith(".xlsx"):
            file_path = os.path.join(data_directory, file)
            df = pd.read_csv(file_path) if file.endswith(".csv") else pd.read_excel(file_path)
            anomalies = detect_anomalies(df, config, case_name)
            print(f"Anomalies in {file} for {case_name}:", anomalies)
