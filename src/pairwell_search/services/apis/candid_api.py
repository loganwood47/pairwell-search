import requests
import time
from typing import List, Dict
from pathlib import Path
import json
from .. import db
from src.pairwell_search.services.embedding_service import embed_texts
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

    API_CALL_COUNTER_FILE = 'src/pairwell_search/services/data_pulls/essentials_api_call_count.txt'

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
        update_api_call_count_in_file(self.API_CALL_COUNTER_FILE)
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
        inserted = db.add_nonprofit(nonprofit_obj)[0]

        if inserted and inserted.get("id"):
            mission = nonprofit_obj.get("mission", "")
            if mission:
                vector = embed_texts([mission])[0]  # embed_texts returns array of arrays
                print("Storing vector for nonprofit ID:", inserted["id"])
                db.store_nonprofit_vector(inserted["id"], vector)
                return [{"status": "inserted", "id": inserted["id"], "message": "Nonprofit and vector added"}]
            return [{"status": "inserted", "message": "Nonprofit added but no mission to embed"}]
        return [{"status": "error", "message": "Failed to add nonprofit"}]
    
    def _seed_nonprofits(self, queries: List[str], max_per_query: int = 50, geo_filter: Dict = None, total_call_cap: int = 100):
        """Uses Candid API, fetch nonprofits for each query and add to DB"""
        for q in queries:
            total_calls = get_api_call_count_from_file(self.API_CALL_COUNTER_FILE)
            if total_calls >= total_call_cap:
                print("API call limit reached, stopping further calls.")
                break
            print("Starting fetch for query:", q)
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



   
class CandidPremierAPI:
    """Client for Candid Premier API"""
    BASE_URL = "https://api.candid.org/premier/v3/"

    API_CALL_COUNTER_FILE = 'src/pairwell_search/services/data_pulls/premier_api_call_count.txt'

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Subscription-Key": self.api_key
        }

    def fetch_nonprofit(self, ein: str) -> dict:
        """Fetch a detailed nonprofit record by EIN"""
        url = f"{self.BASE_URL}/{ein}"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        update_api_call_count_in_file(self.API_CALL_COUNTER_FILE)
        # print(resp.json())
        return resp.json()

    def _transform_premier_record(self, record: dict) -> dict:
        """Flatten Premier response into a simplified schema"""
        summary = record.get("data", {}).get("summary", {})
        financials = record.get("data", {}).get("financials", {}).get("most_recent_year_financials", {})
        programs = record.get("data", {}).get("programs", {}).get("programs", [])
        operations = record.get("data", {}).get("operations", {})

        return {
            # Identity
            "organization_id": summary.get("organization_id"),
            "name": summary.get("organization_name"),
            "ein": summary.get("ein"),
            "aka": summary.get("also_known_as"),
            "mission": summary.get("mission"),
            "year_founded": summary.get("year_founded"),
            "ntee_code": summary.get("ntee_code"),
            "subsection_code": summary.get("subsection_code"),
            "subsection_description": summary.get("subsection_description"),
            # Contact
            "city": summary.get("city"),
            "state": summary.get("state"),
            "zip": summary.get("zip"),
            "latitude": summary.get("latitude"),
            "longitude": summary.get("longitude"),
            "contact_name": summary.get("contact_name"),
            "contact_title": summary.get("contact_title"),
            "contact_email": summary.get("contact_email"),
            "contact_phone": summary.get("contact_phone"),
            "website": summary.get("website_url"),
            "donation_page": summary.get("donation_page"),
            "logo_url": summary.get("logo_url"),
            "social_media": summary.get("social_media_urls", []),
            # Financials (latest year only)
            "fiscal_year": financials.get("fiscal_year"),
            "total_revenue": financials.get("total_revenue"),
            "total_expenses": financials.get("expenses_total"),
            "total_assets": financials.get("assets_total"),
            "net_gain_loss": financials.get("net_gain_loss"),
            "months_of_cash": financials.get("months_of_cash"),
            # Programs (just names + descriptions for now)
            "programs": [
                {
                    "name": p.get("name"),
                    "description": p.get("description"),
                    "areas_served": p.get("areas_served", [])
                }
                for p in programs
            ],
            # --- Operations / People ---
            "leader": {
                "name": operations.get("leader_name"),
                "profile": operations.get("leader_profile")
            },
            "co_leader": {
                "name": operations.get("co_leader_name"),
                "profile": operations.get("co_leader_profile")
            },
            "no_of_employees": operations.get("no_of_employees"),
            "no_of_volunteers": operations.get("no_of_volunteers"),
            "officers_directors_key_employees": [
                {
                    "name": person.get("name"),
                    "title": person.get("title"),
                    "type": person.get("type", []),
                    "compensation": person.get("compensation"),
                    "other_compensation": person.get("other_compensation"),
                    "hours": person.get("hours")
                }
                for person in operations.get("officers_directors_key_employees", [])
            ],
            "senior_staff": [
                {"name": s.get("name"), "title": s.get("title"), "type": s.get("type")}
                for s in operations.get("senior_staff", [])
            ],
            "other_staff": [
                {"name": s.get("name"), "title": s.get("title"), "type": s.get("type")}
                for s in operations.get("other_staff", [])
            ],
            "contractors": [
                {
                    "name": c.get("name"),
                    "service_type": c.get("service_type"),
                    "compensation": c.get("compensation"),
                    "address": c.get("address")
                }
                for c in operations.get("contractors", [])
            ],
            "board_members": [
                {
                    "name": b.get("name"),
                    "title": b.get("title"),
                    "company": b.get("company", []),
                }
                for b in operations.get("board_of_directors", [])
            ]
        }

    def _add_single_nonprofit(self, record: dict) -> list[dict]:
        """Add nonprofit (Premier) to DB"""
        nonprofit_obj = self._transform_premier_record(record)
        nonprofit_obj = clean_record(nonprofit_obj)

        exists_in_db = self.check_nonprofit_exists_in_db(nonprofit_obj["ein"])
        if exists_in_db:
            return [{"status": "exists", "message": "Nonprofit already exists"}]

        inserted = db.add_nonprofit(nonprofit_obj)[0]

        if inserted and inserted.get("id"):
            mission = nonprofit_obj.get("mission", "")
            if mission:
                vector = embed_texts([mission])[0]
                db.store_nonprofit_vector(inserted["id"], vector)
                return [{"status": "inserted", "id": inserted["id"], "message": "Nonprofit + vector added"}]
            return [{"status": "inserted", "message": "Nonprofit added but no mission to embed"}]

        return [{"status": "error", "message": "Failed to add nonprofit"}]
    
    def export_nonprofits_to_json(self, eins: list[str], output_path: str, delay: float = 6.0):
        """
        Fetch nonprofits by EIN, transform them, and export as one JSON file.

        Args:
            eins: List of EIN strings
            output_path: File path to save the aggregated JSON (e.g. 'data/nonprofits.json')
            delay: Seconds to wait between API calls (for rate limiting)
        """
        all_nonprofits = []

        for ein in eins:
            try:
                print(f"Fetching EIN {ein} ...")
                raw_record = self.fetch_nonprofit(ein)
                transformed = self._transform_premier_record(raw_record)
                all_nonprofits.append(transformed)
            except Exception as e:
                print(f"Error fetching EIN {ein}: {e}")
            time.sleep(delay)  # avoid rate limit issues

        # ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_nonprofits, f, indent=2)

        print(f"Exported {len(all_nonprofits)} nonprofits to {output_path}")
        return output_path
    
    def export_nonprofits_with_checkpoint(
        self, 
        eins: list[str], 
        output_path: str, 
        delay: float = 6.0
    ):
        """
        Fetch nonprofits by EIN, transform them, and save incrementally with checkpointing.
        If the process stops, you can resume without re-fetching completed EINs.

        Args:
            eins: List of EIN strings
            output_path: JSON file path for incremental writes
            delay: Seconds to wait between API calls (for rate limiting)
        """
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Load checkpoint if exists
        completed = []
        if Path(output_path).exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    completed = json.load(f)
            except Exception:
                print("Warning: could not load existing checkpoint, starting fresh.")
                completed = []

        done_eins = {n.get("ein") for n in completed if n.get("ein")}
        print(f"Resuming from checkpoint — {len(done_eins)} already saved.")

        for ein in eins:
            if ein in done_eins:
                print(f"Skipping EIN {ein} (already completed).")
                continue

            try:
                print(f"Fetching EIN {ein} ...")
                raw_record = self.fetch_nonprofit(ein)
                transformed = self._transform_premier_record(raw_record)
                completed.append(transformed)

                # Save progress to disk immediately
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(completed, f, indent=2)

                print(f"Saved checkpoint for EIN {ein}")
            except Exception as e:
                print(f"Error fetching EIN {ein}: {e}")
            time.sleep(delay)  # handle rate limits

        print(f"Finished export — {len(completed)} nonprofits saved to {output_path}")
        return output_path

class CandidPremierJsonDataLoader:
    """Utility to load Premier data from JSON into DB"""
    def __init__(self, premier_json_path: str):
        self.premier_json_path = premier_json_path

    def _transform_board_data(self, ein: str, board_members: list[dict], nonprofit_id: int) -> list[dict]:
        """Transform board members data into a simplified schema"""
        transformed = []
        for member in board_members:
            transformed.append({
                "nonprofit_id": nonprofit_id,
                "ein": ein,
                "name": member.get("name"),
                "title": member.get("title"),
                "company": member.get("company", [])
            })
        return transformed
    
    def _transform_projects_data(self, ein: str, projects: list[dict], nonprofit_id: int) -> list[dict]:
        """Transform programs/projects data into a simplified schema"""
        transformed = []
        for project in projects:
            transformed.append({
                "nonprofit_id": nonprofit_id,
                "ein": ein,
                "name": project.get("name"),
                "description": project.get("description"),
                "areas_served": project.get("areas_served", [])
            })
        return transformed
    
    def _transform_key_employees_data(self, ein: str, employees: list[dict], nonprofit_id: int) -> list[dict]:
        """Transform key employees data into a simplified schema"""
        transformed = []
        for emp in employees:
            transformed.append({
                "nonprofit_id": nonprofit_id,
                "ein": ein,
                "name": emp.get("name"),
                "title": emp.get("title"),
                "type": emp.get("type", []),
                "compensation": emp.get("compensation"),
                "other_compensation": emp.get("other_compensation"),
                "hours": emp.get("hours")
            })
        return transformed
    
    def transform_financials_data(self, ein: str, financials: dict, nonprofit_id: int) -> dict:
        """Transform financials data into a simplified schema"""
        return {
            "nonprofit_id": nonprofit_id,
            "ein": ein,
            "fiscal_year": financials.get("fiscal_year"),
            "total_revenue": financials.get("total_revenue"),
            "total_expenses": financials.get("expenses_total"),
            "total_assets": financials.get("assets_total"),
            "net_gain_loss": financials.get("net_gain_loss"),
            "months_of_cash": float(financials.get("months_of_cash")) if financials.get("months_of_cash") else None,
        }

    def _load_into_db(self) -> int:
        """Load nonprofits from Premier JSON file into DB"""
        with open(self.premier_json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        added_count = 0

        projectsArray = []
        boardArray = []
        keyEmpArray = []
        financialsArray = []

        for record in records:
            ein = record.get("ein")
            nonprofit_id = db.get_nonprofit_by_ein(ein=ein)[0].get("id")
            print(record)
            project_data = self._transform_projects_data(record.get("ein"), record.get("programs", []), nonprofit_id)
            board_data = self._transform_board_data(record.get("ein"), record.get("board_members", []), nonprofit_id)
            key_emp_data = self._transform_key_employees_data(record.get("ein"), record.get("operations", {}).get("officers_directors_key_employees", []), nonprofit_id)
            financials_data = self.transform_financials_data(record.get("ein"), record.get("financials", {}), nonprofit_id)
            projectsArray.extend(project_data)
            boardArray.extend(board_data)
            keyEmpArray.extend(key_emp_data)
            financialsArray.append(financials_data)
            added_count += 1

        if len(projectsArray) > 0:
            db.add_nonprofit_projects(projectsArray)
            print("Added projects:", len(projectsArray))
        if len(boardArray) > 0:
            db.add_nonprofit_board_members(boardArray)
            print("Added board members:", len(boardArray))
        if len(keyEmpArray) > 0:
            db.add_nonprofit_key_employees(keyEmpArray)
            print("Added key employees:", len(keyEmpArray))
        if len(financialsArray) > 0:
            db.add_nonprofit_annual_finances(financialsArray)
            print("Added financial records:", len(financialsArray))
        
            # if result and result[0].get("status") == "inserted":
            #     added_count += 1

        print(f"Loaded {added_count} new nonprofits into the database.")
        return added_count