import requests
from typing import List, Dict
import db

class CandidAPI:
    BASE_URL = "https://api.candid.org/essentials/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
            "Subscription-Key": self.api_key
        }

    def fetch_nonprofit(self, ein: str) -> Dict:
        """Fetch a single nonprofit by EIN"""
        url = f"{self.BASE_URL}/organizations/{ein}"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def search_nonprofits(self, query: str, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Search nonprofits by keyword"""
        url = f"{self.BASE_URL}/organizations"
        params = {
            "q": query,
            "limit": limit,
            "offset": offset
        }
        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("organizations", [])

    def seed_nonprofits(self, queries: List[str], max_per_query: int = 50):
        """Fetch nonprofits for each query and add to DB"""
        for q in queries:
            offset = 0
            while True:
                records = self.search_nonprofits(q, limit=max_per_query, offset=offset)
                if not records:
                    break
                for record in records:
                    nonprofit_obj = self._transform_record(record)
                    db.add_nonprofit(nonprofit_obj)
                offset += len(records)

    def _transform_record(self, record: Dict) -> Dict:
        """Transform API record to match nonprofits schema"""
        return {
            "name": record.get("name"),
            "city": record.get("city"),
            "state": record.get("state"),
            "mission": record.get("mission"),
            "ein": record.get("ein"),
            "website": record.get("website"),
            "employee_count": record.get("employee_count"),
            "total_revenue": record.get("total_revenue"),
            "total_giving": record.get("total_giving"),
        }
