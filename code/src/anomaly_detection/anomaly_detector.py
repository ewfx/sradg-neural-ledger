from transformers import pipeline
from sklearn.ensemble import IsolationForest
import pandas as pd
import logging

def detect_anomalies(df, config, case_name):
    anomaly_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    stat_model = IsolationForest(contamination=0.1)

    anomalies = []
    
    case = next((c for c in config["reconciliation_cases"] if c["case_name"] == case_name), None)
    if not case:
        logging.warning(f"Case {case_name} not found in configuration")
        return anomalies
    
    key_columns = case.get("key_columns", [])
    criteria_columns = case.get("criteria_columns", [])
    
    selected_columns = key_columns + criteria_columns
    if not all(col in df.columns for col in selected_columns):
        logging.warning(f"Skipping case {case_name} due to missing columns")
        return anomalies
    
    df_subset = df[selected_columns].dropna()
    stat_model.fit(df_subset)
    predictions = stat_model.predict(df_subset)
    
    for idx, row in df_subset.iterrows():
        input_text = f"Case: {case_name}, Keys: {', '.join(str(row[col]) for col in key_columns)}, Criteria: {', '.join(str(row[col]) for col in criteria_columns)}"
        result = anomaly_model(input_text, candidate_labels=["NORMAL", "ANOMALY"])
        
        if predictions[idx] == -1 or result['labels'][0] == 'ANOMALY':
            anomalies.append({"row": row.to_dict(), "anomaly": result})
    
    logging.info(f"Detected {len(anomalies)} anomalies for case {case_name}")
    return anomalies
