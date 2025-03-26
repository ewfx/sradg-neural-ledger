from transformers import pipeline
from sklearn.ensemble import IsolationForest
import pandas as pd
import logging
from agentautomation.jira_helper import create_jira_ticket

def detect_anomalies(df, config, case_name):
    anomaly_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    stat_model = IsolationForest(contamination=0.1, random_state=42)

    anomalies = []
    
    case = next((c for c in config["reconciliation_cases"] if c["case_name"] == case_name), None)
    if not case:
        logging.warning(f"Case {case_name} not found in configuration")
        return anomalies

    key_columns = case.get("key_columns", [])
    criteria_columns = case.get("criteria_columns", [])
    historical_columns = case.get("historical_columns", [])
    date_columns = case.get("date_columns", [])

    if not date_columns:
        logging.warning(f"No date column specified for case {case_name}")
        return anomalies
    
    date_column = date_columns[0]  # Assuming a single date column
    selected_columns = list(set(key_columns + criteria_columns + historical_columns + [date_column]))

    # Ensure all required columns exist
    missing_cols = [col for col in selected_columns if col not in df.columns]
    if missing_cols:
        logging.warning(f"Skipping case {case_name} due to missing columns: {missing_cols}")
        return anomalies

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    latest_month = df[date_column].max().month
    df["Month"] = df[date_column].dt.strftime("%Y-%m")  # Extract Year-Month

    df_current = df[df[date_column].dt.month == latest_month].reset_index(drop=True)
    df_previous = df[df[date_column].dt.month < latest_month]

    if df_previous.empty or df_current.empty:
        logging.warning(f"No historical or current data available for {case_name}")
        return anomalies

    group_columns = list(set(historical_columns + ["Account"]))
    
    for keys, df_account in df_current.groupby(group_columns):
        df_account_previous = df_previous[df_previous[group_columns].apply(tuple, axis=1) == keys]

        if df_account_previous.empty:
            continue  # Skip if no historical data

        for col in criteria_columns:
            df_account[col] = pd.to_numeric(df_account[col], errors="coerce")
            df_account_previous[col] = pd.to_numeric(df_account_previous[col], errors="coerce")

        df_numeric = df_account_previous[criteria_columns].dropna()
        if df_numeric.empty:
            continue  # Skip if no valid historical data

        stat_model.fit(df_numeric)
        df_account_valid = df_account[criteria_columns + ["Account", "Primary Account", "Secondary Account", "Month"]].dropna().reset_index()
        predictions = stat_model.predict(df_account_valid[criteria_columns]) if not df_account_valid.empty else []

        for idx, (original_idx, row) in enumerate(df_account_valid.iterrows()):
            if idx >= len(predictions):
                continue  # Prevent index error

            # Handle missing 'Company' safely
            keys_data = {col: row[col] for col in key_columns if col in row}
            historical_data = {col: row[col] for col in historical_columns if col in row}

            input_text = f"Case: {case_name}, Keys: {', '.join(str(row[col]) for col in key_columns if col in row)}, " \
                         f"Criteria: {', '.join(f'{col}: {row[col]}' for col in criteria_columns)}"

            result = anomaly_model(input_text, candidate_labels=["NORMAL", "ANOMALY"])
            llm_label = result['labels'][0]
            confidence = result['scores'][0]

            if predictions[idx] == -1 or llm_label == 'ANOMALY':
                anomalies.append({
                    "row": {
                        "Account": row.get("Account", ""),
                        "Primary Account": row.get("Primary Account", ""),
                        "Secondary Account": row.get("Secondary Account", ""),
                        "Month": row.get("Month", ""),
                        **{col: row[col] for col in criteria_columns if col in row}
                    },
                    "anomaly": {
                        "case": case_name,
                        "keys": keys_data,
                        "historical_data": historical_data,
                        "comment": f"Anomaly detected: Significant deviation from historical trend - Confidence {confidence:.2f}."
                    }
                })
                # create jira ticket
                summary = f"[{case_name}] Anomaly Detected - Account {row.get('Account', '')}"
                description = f"Account: {row.get('Account', '')}\nMonth: {row.get('Month', '')}\nCriteria: {row.to_dict()}"
                ticket_key = create_jira_ticket(summary, description)
                # if has ticket id add it to anomalies response
                if ticket_key:
                    anomalies["anomaly"]["jira_ticket"] = ticket_key

    logging.info(f"Detected {len(anomalies)} anomalies for case {case_name}")
    return anomalies
