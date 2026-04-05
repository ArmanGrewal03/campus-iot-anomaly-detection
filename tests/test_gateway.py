from fastapi.testclient import TestClient


def test_health_ok(gateway_client: TestClient):
    resp = gateway_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") in ("healthy", "ok")
    assert "Gateway" in body.get("service", "") or "API Gateway" in body.get("service", "")

