import os
import importlib.util
import pandas as pd
import numpy as np


def _load_feature_store():
    service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "02_Model_Service", "feature_store.py"))
    spec = importlib.util.spec_from_file_location("feature_store_mod", service_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_feature_store_save_load_and_list(tmp_path):
    # Minimal dataset
    df = pd.DataFrame({
        "service": ["http", "dns", "http", "ssh"],
        "state": ["SF", "SF", "S0", "S0"],
        "proto": ["tcp", "udp", "tcp", "udp"],
        "sbytes": [10, 20, 5, 0],
        "dbytes": [0, 5, 10, 1],
    })
    y = np.array([0, 1, 0, 1])

    fs_mod = _load_feature_store()

    vec = fs_mod.DataVectorizer().fit(df)
    X = vec.transform(df)
    assert X.shape[0] == df.shape[0]

    store_dir = tmp_path / "fs"
    fs = fs_mod.FeatureStore(str(store_dir))
    # Use a simple picklable dummy vectorizer to avoid module path issues
    dummy = {"ok": True}
    fs.save("demo", X, y, dummy)

    names = fs.list_features()
    assert "demo" in names

    X2, y2, vec2 = fs.load("demo")
    assert X2 is not None and y2 is not None and vec2 is not None
    assert X2.shape == X.shape
    assert len(y2) == len(y)


def test_vectorizer_handles_unseen_categories():
    fs_mod = _load_feature_store()
    df_train = pd.DataFrame({
        "service": ["http", "dns"],
        "state": ["SF", "S0"],
        "proto": ["tcp", "udp"],
        "sbytes": [1, 2],
        "dbytes": [0, 1],
    })
    df_new = pd.DataFrame({
        "service": ["smtp"],  # unseen
        "state": ["REJ"],     # unseen
        "proto": ["tcp"],     # seen
        "sbytes": [3],
        "dbytes": [1],
    })
    vec = fs_mod.DataVectorizer().fit(df_train)
    X_new = vec.transform(df_new)
    assert X_new.shape[0] == 1
