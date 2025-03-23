import pandas as pd
import json

def load_config(config_path="config/config.json"):
    """Load configuration settings."""
    with open(config_path, "r") as file:
        return json.load(file)

def read_input_data(file):
    """Reads an uploaded file (in-memory) and preprocesses it."""
    config = load_config()
    
    df = pd.read_csv(file)

    # Ensure required columns exist
    required_columns = [config["primary_key"], config["secondary_key"], config["date_column"]] + config["criteria_columns"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert date column to datetime
    df[config["date_column"]] = pd.to_datetime(df[config["date_column"]])
    
    return df
