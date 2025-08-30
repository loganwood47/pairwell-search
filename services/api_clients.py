"""
api_clients.py
Fetches external nonprofit data (Candid, ProPublica, etc.)
"""

import requests

def fetch_nonprofits_from_propublica(ein: str):
    """Fetch nonprofit info by EIN from ProPublica API"""
    url = f"https://projects.propublica.org/nonprofits/api/v2/organizations/{ein}.json"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json()
    return None


def fetch_nonprofits_from_candid(api_key: str, query: str):
    """Stub: Example for Candid API (replace with actual endpoint)"""
    url = f"https://api.candid.org/v1/nonprofits?search={query}"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    return None
