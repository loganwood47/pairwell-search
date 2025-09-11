import requests
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

class CandidEssentialsAPI:
    # TODO: refactor out different API clients
    BASE_URL = "https://api.candid.org/essentials/v3"

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
            "search_terms": ein
            }
        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()
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
        update_api_call_count_in_file('services/data_pulls/essentials_api_call_count.txt')
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
        exists_in_db = self.check_nonprofit_exists_in_db(nonprofit_obj["ein"])
        if exists_in_db:
            return [{"status": "exists", "message": "Nonprofit already exists in DB"}]
        return db.add_nonprofit(nonprofit_obj)
    
    def _seed_nonprofits(self, queries: List[str], max_per_query: int = 50):
        """Uses Candid API, fetch nonprofits for each query and add to DB"""
        for q in queries:
            offset = 0
            while True:
                records = self.search_nonprofits(q, limit=max_per_query, offset=offset)
                if not records:
                    break
                for record in records:
                    self._add_single_nonprofit(record)
                offset += len(records)


    

# class PremierAPI
# TODO: add Premier API client for more detailed data, search existing EINs in DB first