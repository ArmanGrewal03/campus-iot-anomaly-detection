from fastapi.testclient import TestClient


def test_health_ok(model_service_client: TestClient):
    resp = model_service_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") in ("healthy", "ok")
    assert "Model API" in body.get("service", "")


def test_models_list_ok(model_service_client: TestClient):
    resp = model_service_client.get("/models")
    assert resp.status_code == 200
    # Body can vary depending on local files; just assert JSON returned
    assert isinstance(resp.json(), (dict, list))

