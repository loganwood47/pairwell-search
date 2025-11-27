"""
test_db.py
Minimal tests for db.py
Note: This only checks if functions are callable.
For real DB, you should mock Supabase responses.
"""

from src.pairwell_search.services import db
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

def test_db_module_has_functions():
    assert hasattr(db, "get_nonprofits")
    assert hasattr(db, "save_user")
    assert hasattr(db, "save_user_activity")

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_get_nonprofit_vector(mock_client):
    # mock_client.return_value = MagicMock()
    from src.pairwell_search.services import db
    db.supabase = MagicMock()
    nonprofit_id = 100

    db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
        {"vector": [0.1, 0.2, 0.3]}
    ]

    vector = db.get_nonprofit_vector(nonprofit_id)

    assert vector is not None
    assert isinstance(vector, list)
    assert vector == [0.1, 0.2, 0.3]

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_get_nonprofit_vector_nonexistent(mock_client):
    mock_client.return_value = MagicMock()
    nonprofit_id = -1

    db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []

    vector = db.get_nonprofit_vector(nonprofit_id)

    assert vector is None


def test_get_nonprofits_by_id():
    db.supabase = MagicMock()
    nonprofit_ids = [1, 2, 3]
    db.supabase.table.return_value.select.return_value.in_.return_value.limit.return_value.execute.return_value = SimpleNamespace(data=[
            {"id": 1}, {"id": 2}, {"id": 3}
        ])

    nonprofits = db.get_nonprofits_by_id(ids=nonprofit_ids)
    assert nonprofits is not None
    assert isinstance(nonprofits, list)
    assert all('id' in nonprofit for nonprofit in nonprofits)

def test_get_nonprofits_by_id_nonexistent():
    db.supabase = MagicMock()
    nonprofit_ids = [-1, -2]
    db.supabase.table.return_value.select.return_value.in_.return_value.limit.return_value.execute.return_value = SimpleNamespace(data=[])

    nonprofits = db.get_nonprofits_by_id(ids=nonprofit_ids)
    assert nonprofits == []

def test_get_nonprofit_by_ein():
    db.supabase = MagicMock()
    ein = "88-4183627"
    db.supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = SimpleNamespace(data=[
            {"ein": ein}
        ])

    nonprofit = db.get_nonprofit_by_ein(ein=ein)
    assert nonprofit is not None
    assert isinstance(nonprofit, list)
    assert all('ein' in item for item in nonprofit)

def test_get_nonprofit_by_ein_nonexistent():
    db.supabase = MagicMock()
    ein = "000000000"  # Nonexistent EIN
    db.supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = SimpleNamespace(data=[])

    nonprofit = db.get_nonprofit_by_ein(ein=ein)
    assert nonprofit == []

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_get_nonprofit_key_employees(mock_client):
    mock_client.return_value = MagicMock()
    nonprofit_id = 1
    db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = SimpleNamespace(data=[
        {"id": 1, "name": "John Doe", "role": "Director"},
        {"id": 2, "name": "Jane Smith", "role": "Manager"}
    ])

    key_employees = db.get_nonprofit_key_employees(nonprofit_id)

    assert key_employees is not None
    assert isinstance(key_employees, list)
    assert len(key_employees) == 2
    assert all('id' in employee for employee in key_employees)
    assert all('name' in employee for employee in key_employees)
    assert all('role' in employee for employee in key_employees)

def test_get_nonprofit_key_employees_nonexistent():
    db.supabase = MagicMock()
    nonprofit_id = -1
    db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = SimpleNamespace(data=[])

    key_employees = db.get_nonprofit_key_employees(nonprofit_id)

    assert key_employees == []

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_get_nonprofit_projects(mock_client):
    mock_client.return_value = MagicMock()
    nonprofit_id = 1
    db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = SimpleNamespace(data=[
        {"id": 1, "name": "Project A"},
        {"id": 2, "name": "Project B"}
    ])

    projects = db.get_nonprofit_projects(nonprofit_id)

    assert projects is not None
    assert isinstance(projects, list)
    assert len(projects) == 2
    assert all('id' in project for project in projects)
    assert all('name' in project for project in projects)

def test_get_nonprofit_projects_nonexistent():
    db.supabase = MagicMock()
    nonprofit_id = -1
    db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = SimpleNamespace(data=[])

    projects = db.get_nonprofit_projects(nonprofit_id)

    assert projects == []

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_get_nonprofit_board_members(mock_client):
    mock_client.return_value = MagicMock()
    nonprofit_id = 1
    db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = SimpleNamespace(data=[
        {"id": 1, "name": "Alice Johnson", "role": "Chair"},
        {"id": 2, "name": "Bob Smith", "role": "Treasurer"}
    ])

    board_members = db.get_nonprofit_board_members(nonprofit_id)

    assert board_members is not None
    assert isinstance(board_members, list)
    assert len(board_members) == 2
    assert all('id' in member for member in board_members)
    assert all('name' in member for member in board_members)
    assert all('role' in member for member in board_members)

def test_get_nonprofit_board_members_nonexistent():
    db.supabase = MagicMock()
    nonprofit_id = -1
    db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = SimpleNamespace(data=[])

    board_members = db.get_nonprofit_board_members(nonprofit_id)

    assert board_members == []

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_get_nonprofit_financials(mock_client):
    mock_client.return_value = MagicMock()
    nonprofit_id = 1
    db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = SimpleNamespace(data=[
        {"nonprofit_id": nonprofit_id, "financial_data": "sample_data"}
    ])

    financials = db.get_nonprofit_financials(nonprofit_id)

    assert financials is not None
    assert isinstance(financials, list)
    assert len(financials) == 1
    assert all('nonprofit_id' in item for item in financials)

def test_get_nonprofit_financials_nonexistent():
    db.supabase = MagicMock()
    nonprofit_id = -1
    db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = SimpleNamespace(data=[])

    financials = db.get_nonprofit_financials(nonprofit_id)

    assert financials == []

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_add_nonprofit_projects(mock_client):
    mock_client.return_value = MagicMock()
    projects = [{"name": "Project A"}, {"name": "Project B"}]
    db.supabase.table.return_value.insert.return_value.execute.return_value.data = projects

    result = db.add_nonprofit_projects(projects)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2
    assert all('name' in project for project in result)

def test_add_nonprofit_projects_empty():
    db.supabase = MagicMock()
    projects = []
    db.supabase.table.return_value.insert.return_value.execute.return_value.data = []

    result = db.add_nonprofit_projects(projects)

    assert result == []

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_add_nonprofit_key_employees_success(mock_client):
    mock_client.return_value = MagicMock()
    employees = [{"name": "John Doe", "role": "Director"}, {"name": "Jane Smith", "role": "Manager"}]
    db.supabase.table.return_value.insert.return_value.execute.return_value.data = employees

    result = db.add_nonprofit_key_employees(employees)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2
    assert all('name' in employee for employee in result)
    assert all('role' in employee for employee in result)

def test_add_nonprofit_key_employees_empty():
    db.supabase = MagicMock()
    employees = []
    db.supabase.table.return_value.insert.return_value.execute.return_value.data = []

    result = db.add_nonprofit_key_employees(employees)

    assert result == []

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_add_nonprofit_board_members_success(mock_client):
    mock_client.return_value = MagicMock()
    members = [{"name": "Alice Johnson", "role": "Chair"}, {"name": "Bob Smith", "role": "Treasurer"}]
    db.supabase.table.return_value.insert.return_value.execute.return_value.data = members

    result = db.add_nonprofit_board_members(members)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2
    assert all('name' in member for member in result)
    assert all('role' in member for member in result)

def test_add_nonprofit_board_members_empty():
    db.supabase = MagicMock()
    members = []
    db.supabase.table.return_value.insert.return_value.execute.return_value.data = []

    result = db.add_nonprofit_board_members(members)

    assert result == []

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_add_nonprofit_annual_finances_success(mock_client):
    mock_client.return_value = MagicMock()
    finances = [{"nonprofit_id": 1, "financial_data": "sample_data"}]
    db.supabase.table.return_value.insert.return_value.execute.return_value.data = finances

    result = db.add_nonprofit_annual_finances(finances)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert all('nonprofit_id' in finance for finance in result)
    assert all('financial_data' in finance for finance in result)

def test_add_nonprofit_annual_finances_empty():
    db.supabase = MagicMock()
    finances = []
    db.supabase.table.return_value.insert.return_value.execute.return_value.data = []

    result = db.add_nonprofit_annual_finances(finances)

    assert result == []

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_get_np_network_edges_by_id(mock_client):
    mock_client.return_value = MagicMock()
    nonprofit_ids = [1, 2, 3]
    top_k = 15
    db.supabase.table.return_value.select.return_value.in_.return_value.lte.return_value.execute.return_value = SimpleNamespace(data=[
        {"nonprofit_id_a": 1, "nonprofit_id_b": 2, "metadata": {"prox_rank": 10}},
        {"nonprofit_id_a": 2, "nonprofit_id_b": 3, "metadata": {"prox_rank": 5}}
    ])

    edges = db.get_np_network_edges_by_id(nonprofit_id=nonprofit_ids, top_k=top_k)

    assert edges is not None
    assert isinstance(edges, list)
    assert len(edges) == 2
    assert all('nonprofit_id_a' in edge for edge in edges)
    assert all('nonprofit_id_b' in edge for edge in edges)

def test_get_np_network_edges_by_id_nonexistent():
    db.supabase = MagicMock()
    nonprofit_ids = [-1, -2]
    top_k = 15
    db.supabase.table.return_value.select.return_value.in_.return_value.lte.return_value.execute.return_value = SimpleNamespace(data=[])

    edges = db.get_np_network_edges_by_id(nonprofit_id=nonprofit_ids, top_k=top_k)

    assert edges == []

@patch("src.pairwell_search.services.db.get_supabase_client")
def test_fetch_node_attributes(mock_client):
    mock_client.return_value = MagicMock()
    node_ids = [1, 2, 3]
    db.supabase.table.return_value.select.return_value.in_.return_value.execute.return_value = SimpleNamespace(data=[
        {"id": 1, "name": "Nonprofit A", "mission": "Mission A", "total_revenue": 100000, "ntee_codes": ["A", "B"], "logo_url": "url_a", "embedding": [0.1, 0.2]},
        {"id": 2, "name": "Nonprofit B", "mission": "Mission B", "total_revenue": 200000, "ntee_codes": ["C"], "logo_url": "url_b", "embedding": [0.3, 0.4]},
        {"id": 3, "name": "Nonprofit C", "mission": "Mission C", "total_revenue": 300000, "ntee_codes": ["D", "E"], "logo_url": "url_c", "embedding": [0.5, 0.6]}
    ])

    attributes = db.fetch_node_attributes(node_ids)

    assert attributes is not None
    assert isinstance(attributes, dict)
    assert len(attributes) == 3
    assert all(id in attributes for id in node_ids)
    assert attributes[1]["name"] == "Nonprofit A"
    assert attributes[2]["total_revenue"] == 200000
    assert attributes[3]["ntee_codes"] == ["D", "E"]

def test_fetch_node_attributes_nonexistent():
    db.supabase = MagicMock()
    node_ids = [-1, -2]
    db.supabase.table.return_value.select.return_value.in_.return_value.execute.return_value = SimpleNamespace(data=[])

    attributes = db.fetch_node_attributes(node_ids)

    assert attributes == {}