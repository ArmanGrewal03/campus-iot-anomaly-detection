from fastapi.testclient import TestClient


def test_health_ok(data_ingestion_client: TestClient):
    resp = data_ingestion_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") in ("healthy", "ok")
    assert body.get("service")


def test_tables_ok(data_ingestion_client: TestClient):
    resp = data_ingestion_client.get("/tables")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "success"
    assert "tables" in body

