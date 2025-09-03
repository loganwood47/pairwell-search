import requests
from typing import List, Dict
from .. import db
# TODO: add graphql client for Candid taxonomy api
# TODO: normalize NP database, split nonprofit into multiple tables (geo, codes, financials etc)

class CandidAPI:
    # TODO: refactor out different API clients
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
        """matching to nonprofits table schema in Supabase"""
        return {
            "city": record["geography"].get("city"),
            "state": record["geography"].get("state"),
            "latitude": record["geography"].get("latitude"),
            "longitude": record["geography"].get("longitude"),
            "name": record["organization"].get("organization_name"),
            "mission": record["organization"].get("mission"),
            "ein": record["organization"].get("ein"),
            "website": record["organization"].get("website_url"),
            "donation_page": record["organization"].get("donation_page"),
            "contact_email": record["organization"].get("contact_email"),
            "contact_phone": record["organization"].get("contact_phone"),
            "employee_count": record["organization"].get("number_of_employees"),
            "total_revenue": record["financials"]["most_recent_year"].get("total_revenue"),
            "total_expenses": record["financials"]["most_recent_year"].get("total_expenses"),
            "total_assets": record["financials"]["most_recent_year"].get("total_assets"),
            "subject_codes": record["taxonomies"].get("subject_codes"),
            "population_served_codes": record["taxonomies"].get("population_served_codes"),
            "ntee_codes": record["taxonomies"].get("ntee_codes"),
            "subsection_code": record["taxonomies"].get("subsection_code"),
            "foundation_code": record["taxonomies"].get("foundation_code")
        }
    
    def _add_single_nonprofit(self, record: Dict) -> Dict:
        """Add a single nonprofit to the DB"""
        nonprofit_obj = self._transform_record(record)
        return db.add_nonprofit(nonprofit_obj)
