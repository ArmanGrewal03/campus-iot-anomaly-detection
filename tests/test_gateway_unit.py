import importlib.util
import os


def _load_gateway():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "05_Gateway_Proxy", "gateway.py"))
    spec = importlib.util.spec_from_file_location("gateway_mod_unit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_parse_upstream_list_basic():
    gw = _load_gateway()
    assert gw._parse_upstream_list("") == []
    assert gw._parse_upstream_list(" http://a ") == ["http://a"]
    assert gw._parse_upstream_list("http://a, http://b") == ["http://a", "http://b"]


def test_get_target_service_routing():
    gw = _load_gateway()
    # These calls should not raise and should return a string URL
    assert isinstance(gw.get_target_service("/health/data"), str)
    assert isinstance(gw.get_target_service("/predict"), str)
    assert isinstance(gw.get_target_service("/users"), str)
    assert isinstance(gw.get_target_service("/anything-else"), str)

