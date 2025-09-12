import unittest
from unittest.mock import patch, MagicMock
from services.apis.candid_api import CandidEssentialsAPI

class TestCandidEssentialsAPI(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_api_key"
        self.api = CandidEssentialsAPI(self.api_key)

    @patch('services.apis.candid_api.requests.get')
    def test_fetch_nonprofit_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"ein": "123456789"}}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = self.api.fetch_nonprofit("123456789")
        self.assertEqual(result, {"data": {"ein": "123456789"}})
        mock_get.assert_called_once_with(f"{self.api.BASE_URL}/nonprofits/123456789", headers={"Authorization": f"Bearer {self.api_key}"})

    @patch('services.apis.candid_api.requests.get')
    def test_fetch_nonprofit_not_found(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = self.api.fetch_nonprofit("invalid_ein")
        self.assertIsNone(result)

    @patch('services.apis.candid_api.requests.get')
    def test_search_nonprofits_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"ein": "123456789"}, {"ein": "987654321"}]}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = self.api.search_nonprofits("test query")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['ein'], "123456789")

    @patch('services.apis.candid_api.requests.get')
    def test_search_nonprofits_empty_result(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = self.api.search_nonprofits("nonexistent query")
        self.assertEqual(result, [])

    def test_check_nonprofit_exists_in_db(self):
        # Assuming a mock database check
        with patch('services.apis.candid_api.db.get_nonprofit_by_ein') as mock_check:
            mock_check.return_value = [True]
            result = self.api.check_nonprofit_exists_in_db("123456789")
            self.assertTrue(result)

    def test_transform_record(self):
        record = {"ein": "123456789", "name": "Test Nonprofit"}
        transformed = self.api._transform_record(record)
        self.assertEqual(transformed['ein'], "123456789")
        self.assertEqual(transformed['name'], "Test Nonprofit")

if __name__ == '__main__':
    unittest.main()