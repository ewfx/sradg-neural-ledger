import pandas as pd
from io import BytesIO

def process_uploaded_file(file):
    ext = file.name.split('.')[-1].lower()
    if ext == "csv":
        df = pd.read_csv(file)
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")
    return df
