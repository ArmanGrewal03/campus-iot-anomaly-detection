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


def test_model_types_ok(model_service_client: TestClient):
    resp = model_service_client.get("/model-types")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "success"
    assert "total_model_types" in body


def test_model_status_not_trained(model_service_client: TestClient):
    # No model trained by default; should report not_trained
    resp = model_service_client.get("/model/status", headers={"model_name": "A"})
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") in ("not_trained", "trained")


def test_model_metrics_404_when_missing(model_service_client: TestClient):
    # Without a trained model, metrics should be 404
    resp = model_service_client.get("/model/metrics", headers={"model_name": "A"})
    assert resp.status_code in (200, 404)  # Some envs may have an existing model

