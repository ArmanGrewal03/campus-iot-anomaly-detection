from fastapi.testclient import TestClient


def test_health_ok(user_service_client: TestClient):
    resp = user_service_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") in ("healthy", "ok")
    assert "User Service" in body.get("service", "") or "User" in body.get("service", "")

