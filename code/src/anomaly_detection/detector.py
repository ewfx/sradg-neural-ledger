import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    df["abs_diff"] = abs(df["criteria_col_1"] - df["criteria_col_2"])
    df["rolling_mean"] = df["abs_diff"].rolling(window=10).mean()
    df["rolling_std"] = df["abs_diff"].rolling(window=10).std()
    df["upper_band"] = df["rolling_mean"] + (2 * df["rolling_std"])
    df["lower_band"] = df["rolling_mean"] - (2 * df["rolling_std"])
    df["anomaly"] = (df["abs_diff"] > df["upper_band"]) | (df["abs_diff"] < df["lower_band"])

    iso_forest = IsolationForest(contamination=0.05)
    df["anomaly_ml"] = iso_forest.fit_predict(df[["abs_diff"]])

    anomalies = df[df["anomaly"] == True]
    return anomalies.to_dict(orient="records")
