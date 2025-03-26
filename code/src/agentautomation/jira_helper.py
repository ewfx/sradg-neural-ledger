import requests
import logging

JIRA_URL = ""
JIRA_EMAIL = ""
# provide pat token
JIRA_API_TOKEN = ""
# provide project key
JIRA_PROJECT_KEY = ""

def create_jira_ticket(summary, description):
    """Create a Jira issue via API."""
    url = f"{JIRA_URL}/rest/api/2/issue"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    auth = (JIRA_EMAIL, JIRA_API_TOKEN)

    data = {
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Task"}
        }
    }

    response = requests.post(url, json=data, headers=headers, auth=auth)

    if response.status_code == 201:
        ticket_key = response.json().get("key")
        logging.info(f"Jira ticket created: {ticket_key}")
        return ticket_key
    else:
        logging.error(f"Failed to create Jira ticket: {response.text}")
        return None
