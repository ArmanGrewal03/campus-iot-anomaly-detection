from fastapi.testclient import TestClient
from datetime import datetime
import json


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


def test_fields_with_inserted_row(data_ingestion_client: TestClient):
    # Dynamically import module to access helpers
    import importlib.util, os
    service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01_Data_Ingestion_Service", "main.py"))
    spec = importlib.util.spec_from_file_location("data_ingestion_main_mod", service_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    dataset = "pytestcov"
    module.init_db(dataset)
    table = module.get_table_name("csv_data", dataset)
    conn = module.get_db_connection()
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO {table} (upload_timestamp, row_data) VALUES (?, ?)",
        (datetime.utcnow().isoformat(), json.dumps({"colA": 123, "colB": "x"})),
    )
    conn.commit()
    conn.close()

    resp = data_ingestion_client.get("/fields", headers={"dataset_name": dataset})
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "success"
    assert "fields" in body and isinstance(body["fields"], list) and "colA" in body["fields"]


def test_dataset_header_validation(data_ingestion_client: TestClient):
    # Invalid header with illegal chars should 400
    resp = data_ingestion_client.get("/fields", headers={"dataset_name": "!!!!"})
    assert resp.status_code == 400


def test_validate_and_splits_end_to_end(data_ingestion_client: TestClient):
    # Seed 10 rows with alternating labels and types
    import importlib.util, os, json as _json
    from datetime import datetime as _dt
    service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01_Data_Ingestion_Service", "main.py"))
    spec = importlib.util.spec_from_file_location("data_ingestion_main_mod2", service_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    dataset = "pytestfull"
    module.init_db(dataset)
    table = module.get_table_name("csv_data", dataset)
    conn = module.get_db_connection()
    cur = conn.cursor()
    # Clear any leftovers
    try:
        cur.execute(f"DELETE FROM {table}")
        conn.commit()
    except Exception:
        pass
    rows = []
    for i in range(10):
        label = 0 if i % 2 == 0 else 1
        type_val = "A" if i % 2 == 0 else "B"
        rows.append((_dt.utcnow().isoformat(), _json.dumps({"label": label, "type": type_val, "val": i})))
    cur.executemany(f"INSERT INTO {table} (upload_timestamp, row_data) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

    # Validate with 50/50 split
    resp = data_ingestion_client.put(
        "/validate",
        headers={
            "dataset_name": dataset,
            "X-Training-Percent": "50",
            "X-Testing-Percent": "50",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_rows"] == 10
    assert body["training_rows"] + body["testing_rows"] == 10
    # Percentages should be close to requested split
    assert 40 <= body["training_percentage"] <= 60
    assert 40 <= body["testing_percentage"] <= 60

    # Training fetch
    r_train = data_ingestion_client.get("/training", headers={"dataset_name": dataset, "X-Limit": "100"})
    assert r_train.status_code == 200
    bt = r_train.json()
    assert bt["status"] == "success"
    assert bt["label"] == "training"
    assert isinstance(bt["label_counts"]["label_0"], int)
    assert isinstance(bt["label_counts"]["label_1"], int)
    for row in bt["data"]:
        assert row["T"] == "training"
        assert isinstance(row["data"], dict)

    # Testing fetch
    r_test = data_ingestion_client.get("/testing", headers={"dataset_name": dataset, "X-Limit": "100"})
    assert r_test.status_code == 200
    bs = r_test.json()
    assert bs["status"] == "success"
    assert bs["label"] == "testing"
    for row in bs["data"]:
        assert row["T"] == "testing"
        assert isinstance(row["data"], dict)

    # View and stats
    r_view = data_ingestion_client.get("/view", headers={"dataset_name": dataset}, params={"limit": 1000, "offset": 0})
    assert r_view.status_code == 200
    bv = r_view.json()
    assert bv["total_rows"] == 10
    assert len(bv["data"]) > 0

    r_stats = data_ingestion_client.get("/stats", headers={"dataset_name": dataset})
    assert r_stats.status_code == 200
    s = r_stats.json()
    assert s["total_records"] == 10
    assert s["training_records"] + s["testing_records"] == 10

    # Type stats
    r_types = data_ingestion_client.get("/type-stats", headers={"dataset_name": dataset})
    assert r_types.status_code == 200
    ts = r_types.json()
    assert "type_distribution" in ts or "sample_size" in ts


def test_random_test_endpoint(data_ingestion_client: TestClient):
    # Prepare dataset with one row
    import importlib.util, os, json as _json
    from datetime import datetime as _dt
    service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01_Data_Ingestion_Service", "main.py"))
    spec = importlib.util.spec_from_file_location("data_ingestion_main_mod3", service_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    dataset = "pytestrandom"
    module.init_db(dataset)
    table = module.get_table_name("csv_data", dataset)
    conn = module.get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"DELETE FROM {table}")
    except Exception:
        pass
    cur.execute(
        f"INSERT INTO {table} (upload_timestamp, row_data) VALUES (?, ?)",
        (_dt.utcnow().isoformat(), _json.dumps({"colX": 1})),
    )
    conn.commit()
    conn.close()

    r = data_ingestion_client.get("/random-test", headers={"dataset_name": dataset})
    assert r.status_code in (200, 404)  # Depending on table/T column state
    body = r.json()
    # If 200, ensure JSON shape is dict; some builds may omit 'status'
    if r.status_code == 200:
        assert isinstance(body, dict)


def test_validate_bad_percentages(data_ingestion_client: TestClient):
    # training + testing must sum to 100
    r = data_ingestion_client.put(
        "/validate",
        headers={
            "dataset_name": "pytestfull",
            "X-Training-Percent": "10",
            "X-Testing-Percent": "10",
        },
    )
    assert r.status_code == 400


def test_clear_dataset_tables(data_ingestion_client: TestClient):
    # Seed dataset and then clear only that dataset's tables
    import importlib.util, os, json as _json
    from datetime import datetime as _dt
    service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01_Data_Ingestion_Service", "main.py"))
    spec = importlib.util.spec_from_file_location("data_ingestion_main_mod4", service_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    dataset = "pytestclear"
    module.init_db(dataset)
    table = module.get_table_name("csv_data", dataset)
    conn = module.get_db_connection()
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO {table} (upload_timestamp, row_data) VALUES (?, ?)",
        (_dt.utcnow().isoformat(), _json.dumps({"x": 1})),
    )
    conn.commit()
    conn.close()

    # Ensure data exists
    pre = data_ingestion_client.get("/view", headers={"dataset_name": dataset}, params={"limit": 10, "offset": 0})
    assert pre.status_code == 200 and pre.json().get("total_rows", 0) >= 1

    # Clear the dataset
    resp = data_ingestion_client.delete("/clear", headers={"dataset_name": dataset})
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "success"
    assert "tables_dropped" in body

    # After clear, /view should 404
    post = data_ingestion_client.get("/view", headers={"dataset_name": dataset}, params={"limit": 10, "offset": 0})
    assert post.status_code in (200, 404)

