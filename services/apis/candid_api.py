import requests
import time
from typing import List, Dict
from .. import db
# TODO: add graphql client for Candid taxonomy api
# TODO: normalize NP database, split nonprofit into multiple tables (geo, codes, financials etc)

# Utility to track API call count in a file
def update_api_call_count_in_file(filename: str) -> int:
    """Utility to track API call count in a file"""
    try:
        with open(filename, 'r') as f:
            count = int(f.read().strip())
    except FileNotFoundError:
        count = 0
    count += 1
    with open(filename, 'w') as f:
        f.write(str(count))
    return count

def get_api_call_count_from_file(filename: str) -> int:
    """Utility to get current API call count from a file"""
    try:
        with open(filename, 'r') as f:
            count = int(f.read().strip())
    except FileNotFoundError:
        count = 0
    return count

def clean_record(record: dict) -> dict:
    """Recursively clean dict: convert empty strings to None."""
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, str) and v.strip() == "":
            cleaned[k] = None
        elif isinstance(v, dict):
            cleaned[k] = clean_record(v)
        elif isinstance(v, list):
            cleaned[k] = [clean_record(i) if isinstance(i, dict) else i for i in v]
        else:
            cleaned[k] = v
    return cleaned


class CandidEssentialsAPI:
    """Client for Candid Essentials API"""
    # TODO: refactor out different API clients
    BASE_URL = "https://api.candid.org/essentials/v3"

    API_CALL_COUNTER_FILE = 'services/data_pulls/essentials_api_call_count.txt'

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-Type": "text/json",
            "accept": "application/json",
            "Subscription-Key": self.api_key
        }

    def fetch_nonprofit(self, ein: str) -> Dict:
        """Fetch a single nonprofit by EIN"""
        url = f"{self.BASE_URL}"
        params = {
            "search_terms": ein,
            "from": 0,
            "size": 1
            }
        resp = requests.post(url, headers=self.headers, json=params)
        resp.raise_for_status()
        update_api_call_count_in_file(self.API_CALL_COUNTER_FILE)
        return resp.json()

    def search_nonprofits(
            self, 
            query: str, 
            limit: int = 25, 
            offset: int = 0, 
            states: list[str] = None,
            metros: list[str] = None,
            cities: list[str] = None,
            counties: list[str] = None,
            zip: str = None,
            radius: int = None
            ) -> List[Dict]:
        
        """Search nonprofits by keyword"""
        url = f"{self.BASE_URL}"

        geography = {
            "state": states or [],
            "msa": metros or [],
            "city": cities or [],
            "county": counties or [],
            "zip": zip,
            "radius": radius
        }

        filters = {
            "geography": {k: v for k, v in geography.items() if v},  # only include non-empty filters
            # TODO: could add organization size, financials, taxonomies etc filters here
        }

        params = {
            # "q": query,
            "search_terms": query,
            "from": offset,
            "size": limit,
            "filters": filters
        }

        resp = requests.post(url, headers=self.headers, json=params)
        resp.raise_for_status()
        data = resp.json()
        update_api_call_count_in_file(self.API_CALL_CALL_COUNTER_FILE)
        return data.get("hits", [])
    
    def check_nonprofit_exists_in_db(self, ein: str) -> bool:
        """Check if a nonprofit exists in the DB by EIN"""
        existing = db.get_nonprofit_by_ein(ein=ein)
        existenceCheck = True if (existing and len(existing) > 0) else False
        print("Existing in DB check for EIN", ein, ":", existenceCheck)
        return existenceCheck

    def _transform_record(self, record: Dict) -> Dict:
        """matching to nonprofits table schema in Supabase"""
        return {
            # Geos
            "city": record["geography"].get("city"),
            "state": record["geography"].get("state"),
            "latitude": record["geography"].get("latitude"),
            "longitude": record["geography"].get("longitude"),
            # Org object
            "name": record["organization"].get("organization_name"),
            "mission": record["organization"].get("mission"),
            "ein": record["organization"].get("ein"),
            "website": record["organization"].get("website_url"),
            "donation_page": record["organization"].get("donation_page"),
            "contact_email": record["organization"].get("contact_email"),
            "contact_phone": record["organization"].get("contact_phone"),
            "employee_count": record["organization"].get("number_of_employees"),
            "logo_url": record["organization"].get("logo_url"),
            # Financials
            "total_revenue": record["financials"]["most_recent_year"].get("total_revenue"),
            "total_expenses": record["financials"]["most_recent_year"].get("total_expenses"),
            "total_assets": record["financials"]["most_recent_year"].get("total_assets"),
            # Taxonomies
            "subject_codes": record["taxonomies"].get("subject_codes"),
            "population_served_codes": record["taxonomies"].get("population_served_codes"),
            "ntee_codes": record["taxonomies"].get("ntee_codes"),
            "subsection_code": record["taxonomies"].get("subsection_code"),
            "foundation_code": record["taxonomies"].get("foundation_code")
        }
    
    def _add_single_nonprofit(self, record: Dict) -> list[Dict]:
        """Add a single nonprofit to the DB"""
        nonprofit_obj = self._transform_record(record)
        nonprofit_obj = clean_record(nonprofit_obj)
        exists_in_db = self.check_nonprofit_exists_in_db(nonprofit_obj["ein"])
        if exists_in_db:
            return [{"status": "exists", "message": "Nonprofit already exists in DB"}]
        print("Adding nonprofit to DB:", nonprofit_obj["name"], "EIN:", nonprofit_obj["ein"])
        return db.add_nonprofit(nonprofit_obj)
    
    def _seed_nonprofits(self, queries: List[str], max_per_query: int = 50, geo_filter: Dict = None):
        """Uses Candid API, fetch nonprofits for each query and add to DB"""
        for q in queries:
            total_calls = get_api_call_count_from_file(self.API_CALL_COUNTER_FILE)
            if total_calls >= 74:
                print("API call limit reached, stopping further calls.")
                break
            offset = 0
            while True:
                print("Fetching nonprofits for query:", q, "offset:", offset)
                records = self.search_nonprofits(
                    query=q, 
                    states=geo_filter.get("states") if geo_filter else None,
                    metros=geo_filter.get("metros") if geo_filter else None,
                    cities=geo_filter.get("cities") if geo_filter else None,
                    counties=geo_filter.get("counties") if geo_filter else None,
                    zip=geo_filter.get("zip") if geo_filter else None,
                    radius=geo_filter.get("radius") if geo_filter else None,
                    limit=max_per_query, 
                    offset=offset)
                time.sleep(6) # rate limit handling, max 10 calls/min
                if not records:
                    break
                for record in records:
                    self._add_single_nonprofit(record)
                offset += len(records)
                if offset >= 75:
                    break


    

# class PremierAPI
# TODO: add Premier API client for more detailed data, search existing EINs in DB first