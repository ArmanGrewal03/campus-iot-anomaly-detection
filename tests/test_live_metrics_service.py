from fastapi.testclient import TestClient


def test_health_ok(live_metrics_client: TestClient):
    resp = live_metrics_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") in ("healthy", "ok")
    assert "live-metrics" in body.get("service", "") or "Live Metrics" in body.get("service", "")


def test_metrics_shape_ok(live_metrics_client: TestClient):
    resp = live_metrics_client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    # Basic structure checks
    for key in ["labels", "request_status", "blocking_status", "query_per_second", "packet_rate"]:
        assert key in body
        assert isinstance(body[key], list)
    # Lengths should align
    n = len(body["labels"])
    assert n > 0
    assert len(body["request_status"]) == n
    assert len(body["blocking_status"]) == n
    assert len(body["query_per_second"]) == n
    assert len(body["packet_rate"]) == n

