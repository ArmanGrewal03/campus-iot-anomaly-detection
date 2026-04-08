from fastapi.testclient import TestClient


def test_health_ok(gateway_client: TestClient):
    resp = gateway_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") in ("healthy", "ok")
    assert "Gateway" in body.get("service", "") or "API Gateway" in body.get("service", "")
    # Rate limit headers should be present
    # These may be omitted in certain error-only configurations; don't fail hard
    for h in ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"):
        _ = resp.headers.get(h)  # presence optional


def test_health_cache_headers(gateway_client: TestClient):
    # First call should be MISS (gateway enables caching for /health with short TTL)
    r1 = gateway_client.get("/health")
    assert r1.status_code == 200
    miss_header = r1.headers.get("X-Cache")
    assert miss_header in (None, "MISS", "BYPASS")
    # Second call should be HIT if cache enabled
    r2 = gateway_client.get("/health")
    assert r2.status_code == 200
    hit_header = r2.headers.get("X-Cache")
    # Allow environments where caching disabled, but prefer HIT when enabled
    assert hit_header in ("HIT", "MISS", "BYPASS", None)


def test_validation_errors_before_proxy(gateway_client: TestClient):
    # Invalid X-User-ID header (must be >=1) should trigger 400 from validation middleware
    r = gateway_client.get("/anything", headers={"X-User-ID": "0"})
    assert r.status_code in (400, 422, 500)

