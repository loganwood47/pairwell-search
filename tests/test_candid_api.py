import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import requests
from services.apis.candid_api import (
    CandidEssentialsAPI,
    update_api_call_count_in_file,
    get_api_call_count_from_file,
    clean_record
)


class TestUtils(unittest.TestCase):

    def test_update_api_call_count_in_file_new_file(self):
        # Simulate FileNotFoundError first
        mock_file = mock_open()
        with patch("builtins.open", side_effect=[FileNotFoundError(), mock_file.return_value]):
            count = update_api_call_count_in_file("fake.txt")
            self.assertEqual(count, 1)

    def test_update_api_call_count_in_file_existing(self):
        m = mock_open(read_data="3")
        with patch("builtins.open", m):
            count = update_api_call_count_in_file("fake.txt")
            self.assertEqual(count, 4)

    def test_get_api_call_count_from_file_new_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError()):
            count = get_api_call_count_from_file("fake.txt")
            self.assertEqual(count, 0)

    def test_get_api_call_count_from_file_existing(self):
        m = mock_open(read_data="7")
        with patch("builtins.open", m):
            count = get_api_call_count_from_file("fake.txt")
            self.assertEqual(count, 7)

    def test_clean_record_nested(self):
        record = {
            "a": "",
            "b": {"c": ""},
            "d": [{"x": ""}, {"y": "val"}],
            "e": "non-empty"
        }
        cleaned = clean_record(record)
        self.assertIsNone(cleaned["a"])
        self.assertIsNone(cleaned["b"]["c"])
        self.assertIsNone(cleaned["d"][0]["x"])
        self.assertEqual(cleaned["d"][1]["y"], "val")
        self.assertEqual(cleaned["e"], "non-empty")


class TestCandidEssentialsAPI(unittest.TestCase):

    def setUp(self):
        self.api_key = "fake_api_key"
        self.api = CandidEssentialsAPI(self.api_key)

    @patch("services.apis.candid_api.requests.post")
    @patch("services.apis.candid_api.update_api_call_count_in_file")
    def test_fetch_nonprofit_success(self, mock_update, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"ein": "123"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = self.api.fetch_nonprofit("123")
        self.assertEqual(result, {"data": {"ein": "123"}})
        mock_post.assert_called_once()
        mock_update.assert_called_once_with(self.api.API_CALL_COUNTER_FILE)

    @patch("services.apis.candid_api.requests.post")
    def test_fetch_nonprofit_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_post.return_value = mock_resp

        with self.assertRaises(requests.exceptions.HTTPError):
            self.api.fetch_nonprofit("bad")

    @patch("services.apis.candid_api.requests.post")
    @patch("services.apis.candid_api.update_api_call_count_in_file")
    def test_search_nonprofits_with_filters(self, mock_update, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"hits": [{"ein": "111"}, {"ein": "222"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = self.api.search_nonprofits(
            query="health", states=["CA"], cities=["LA"], limit=5, offset=0
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["ein"], "111")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        # Ensure filters were included in payload
        self.assertIn("filters", kwargs["json"])
        self.assertIn("geography", kwargs["json"]["filters"])

    @patch("services.apis.candid_api.requests.post")
    @patch("services.apis.candid_api.update_api_call_count_in_file")
    def test_search_nonprofits_empty_hits(self, mock_update, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"hits": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = self.api.search_nonprofits("nothing")
        self.assertEqual(result, [])

    @patch("services.apis.candid_api.db.get_nonprofit_by_ein")
    def test_check_nonprofit_exists_in_db_true(self, mock_db):
        mock_db.return_value = [{"ein": "123"}]
        result = self.api.check_nonprofit_exists_in_db("123")
        self.assertTrue(result)

    @patch("services.apis.candid_api.db.get_nonprofit_by_ein")
    def test_check_nonprofit_exists_in_db_false(self, mock_db):
        mock_db.return_value = []
        result = self.api.check_nonprofit_exists_in_db("notthere")
        self.assertFalse(result)

    def test_transform_record(self):
        record = {
            "geography": {"city": "LA", "state": "CA", "latitude": 34, "longitude": -118},
            "organization": {
                "organization_name": "Org",
                "mission": "Do good",
                "ein": "999",
                "website_url": "http://org.org",
                "donation_page": "http://donate.org",
                "contact_email": "test@org.org",
                "contact_phone": "555-555",
                "number_of_employees": 10,
                "logo_url": "http://logo.png"
            },
            "financials": {"most_recent_year": {
                "total_revenue": 1000,
                "total_expenses": 500,
                "total_assets": 2500
            }},
            "taxonomies": {
                "subject_codes": ["S1"],
                "population_served_codes": ["P1"],
                "ntee_codes": ["N1"],
                "subsection_code": "SC",
                "foundation_code": "FC"
            }
        }
        transformed = self.api._transform_record(record)
        self.assertEqual(transformed["ein"], "999")
        self.assertEqual(transformed["city"], "LA")
        self.assertEqual(transformed["total_revenue"], 1000)

    @patch("services.apis.candid_api.db.add_nonprofit")
    @patch("services.apis.candid_api.CandidEssentialsAPI.check_nonprofit_exists_in_db")
    def test_add_single_nonprofit_exists(self, mock_check, mock_add):
        record = {
            "geography": {"city": "LA", "state": "CA", "latitude": 34, "longitude": -118},
            "organization": {
                "organization_name": "Org",
                "mission": "Do good",
                "ein": "999",
                "website_url": "http://org.org",
                "donation_page": "http://donate.org",
                "contact_email": "test@org.org",
                "contact_phone": "555-555",
                "number_of_employees": 10,
                "logo_url": "http://logo.png"
            },
            "financials": {"most_recent_year": {
                "total_revenue": 1000,
                "total_expenses": 500,
                "total_assets": 2500
            }},
            "taxonomies": {
                "subject_codes": ["S1"],
                "population_served_codes": ["P1"],
                "ntee_codes": ["N1"],
                "subsection_code": "SC",
                "foundation_code": "FC"
            }
        }
        mock_check.return_value = True
        result = self.api._add_single_nonprofit(record)
        self.assertEqual(result[0]["status"], "exists")
        mock_add.assert_not_called()

    @patch("services.apis.candid_api.db.add_nonprofit")
    @patch("services.apis.candid_api.CandidEssentialsAPI.check_nonprofit_exists_in_db")
    @patch("services.apis.candid_api.clean_record")
    @patch("services.apis.candid_api.CandidEssentialsAPI._transform_record")
    def test_add_single_nonprofit_exists(self, mock_transform, mock_clean, mock_check, mock_add):
        mock_transform.return_value = {"ein": "123", "name": "Test Org"}
        mock_clean.return_value = {"ein": "123", "name": "Test Org"}
        mock_check.return_value = True  # already exists

        result = self.api._add_single_nonprofit({"org": "raw"})
        self.assertEqual(result, [{"status": "exists", "message": "Nonprofit already exists in DB"}])
        mock_add.assert_not_called()

    @patch("services.apis.candid_api.db.add_nonprofit")
    @patch("services.apis.candid_api.CandidEssentialsAPI.check_nonprofit_exists_in_db")
    @patch("services.apis.candid_api.clean_record")
    @patch("services.apis.candid_api.CandidEssentialsAPI._transform_record")
    def test_add_single_nonprofit_new_no_mission(self, mock_transform, mock_clean, mock_check, mock_add):
        mock_transform.return_value = {"ein": "123", "name": "Test Org"}
        mock_clean.return_value = {"ein": "123", "name": "Test Org"}  # no mission field
        mock_check.return_value = False
        mock_add.return_value = [{"id": "abc123", "ein": "123", "name": "Test Org"}]

        result = self.api._add_single_nonprofit({"org": "raw"})
        self.assertEqual(
            result,
            [{"status": "inserted", "message": "Nonprofit added but no mission to embed"}],
        )
        mock_add.assert_called_once()

    @patch("services.apis.candid_api.db.store_nonprofit_vector")
    @patch("services.apis.candid_api.embed_texts")
    @patch("services.apis.candid_api.db.add_nonprofit")
    @patch("services.apis.candid_api.CandidEssentialsAPI.check_nonprofit_exists_in_db")
    @patch("services.apis.candid_api.clean_record")
    @patch("services.apis.candid_api.CandidEssentialsAPI._transform_record")
    def test_add_single_nonprofit_new_with_mission(
        self, mock_transform, mock_clean, mock_check, mock_add, mock_embed, mock_store
    ):
        mock_transform.return_value = {"ein": "123", "name": "Test Org", "mission": "Do good"}
        mock_clean.return_value = {"ein": "123", "name": "Test Org", "mission": "Do good"}
        mock_check.return_value = False
        mock_add.return_value = [{"id": "abc123", "ein": "123", "name": "Test Org", "mission": "Do good"}]

        mock_embed.return_value = [[0.1, 0.2, 0.3]]  # pretend embedding
        mock_store.return_value = {"status": "ok"}

        result = self.api._add_single_nonprofit({"org": "raw"})
        self.assertEqual(
            result,
            [{"status": "inserted", "id": "abc123", "message": "Nonprofit and vector added"}],
        )
        mock_add.assert_called_once()
        mock_embed.assert_called_once_with(["Do good"])
        mock_store.assert_called_once_with("abc123", [0.1, 0.2, 0.3])

    @patch("services.apis.candid_api.get_api_call_count_from_file")
    @patch("services.apis.candid_api.CandidEssentialsAPI.search_nonprofits")
    @patch("services.apis.candid_api.CandidEssentialsAPI._add_single_nonprofit")
    @patch("services.apis.candid_api.time.sleep", return_value=None)  # no waiting
    def test_seed_nonprofits_respects_cap(self, mock_sleep, mock_add, mock_search, mock_get_count):
        # Cap already reached
        mock_get_count.return_value = 100
        self.api._seed_nonprofits(["query1"], total_call_cap=100)
        mock_search.assert_not_called()

    @patch("services.apis.candid_api.get_api_call_count_from_file")
    @patch("services.apis.candid_api.CandidEssentialsAPI.search_nonprofits")
    @patch("services.apis.candid_api.CandidEssentialsAPI._add_single_nonprofit")
    @patch("services.apis.candid_api.time.sleep", return_value=None)
    def test_seed_nonprofits_adds_records(self, mock_sleep, mock_add, mock_search, mock_get_count):
        mock_get_count.return_value = 0
        mock_search.side_effect = [
            [{"ein": "1"}, {"ein": "2"}],  # first batch
            []  # second batch ends loop
        ]
        self.api._seed_nonprofits(["query1"], max_per_query=2, total_call_cap=5)
        mock_add.assert_has_calls([call({"ein": "1"}), call({"ein": "2"})])
        self.assertEqual(mock_search.call_count, 2)


if __name__ == "__main__":
    unittest.main()
