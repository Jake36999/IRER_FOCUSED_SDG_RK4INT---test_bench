import pytest
import sqlite3
from aste_hunter import insert_simulation_result, pareto_select

@pytest.fixture
def db_connection(tmp_path):
    db = tmp_path / "test_simulation_ledger.db"
    conn = sqlite3.connect(db)
    yield conn
    conn.close()

# Test DB insert
def test_insert_simulation_result(db_connection):
    result = {"sse": 0.01, "pcs": 0.9, "ic": 0.8, "parent_1": 1, "parent_2": 2}
    insert_simulation_result(db_connection, result)
    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM simulation_ledger")
    rows = cursor.fetchall()
    assert len(rows) == 1

# Test Pareto selection
def test_pareto_select():
    population = [
        {"sse": 0.01, "pcs": 0.9, "ic": 0.8},
        {"sse": 0.02, "pcs": 0.95, "ic": 0.7},
        {"sse": 0.005, "pcs": 0.85, "ic": 0.9}
    ]
    selected = pareto_select(population)
    assert isinstance(selected, list)
    assert all("sse" in ind for ind in selected)
