import os
import sys
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Load apps via absolute file paths because service folders start with digits
import importlib.util


@pytest.fixture(scope="session")
def data_ingestion_client() -> Generator[TestClient, None, None]:
    service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01_Data_Ingestion_Service", "main.py"))
    spec = importlib.util.spec_from_file_location("data_ingestion_main", service_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    with TestClient(module.app) as client:
        yield client


@pytest.fixture(scope="session")
def model_service_client() -> Generator[TestClient, None, None]:
    service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "02_Model_Service", "model_api.py"))
    spec = importlib.util.spec_from_file_location("model_service_api", service_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    with TestClient(module.app) as client:
        yield client


@pytest.fixture(scope="session")
def user_service_client() -> Generator[TestClient, None, None]:
    service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_User_Service", "user_service.py"))
    spec = importlib.util.spec_from_file_location("user_service", service_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    with TestClient(module.app) as client:
        yield client


@pytest.fixture(scope="session")
def gateway_client() -> Generator[TestClient, None, None]:
    service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "05_Gateway_Proxy", "gateway.py"))
    spec = importlib.util.spec_from_file_location("gateway_service", service_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    # Avoid re-raising server exceptions from middleware validation
    with TestClient(module.app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture(scope="session")
def live_metrics_client() -> Generator[TestClient, None, None]:
    service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "06_Live_Metrics_Service", "live_metrics_service.py"))
    spec = importlib.util.spec_from_file_location("live_metrics_service", service_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    with TestClient(module.app) as client:
        yield client

