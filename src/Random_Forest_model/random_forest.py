# random_forest.py
import json, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Your exact features (+ encoded protocol)
FEATURES = ["bytes_in","bytes_out","packets_in","packets_out","duration_s","protocol_enc"]

# Versions & settings
MODEL_VERSION = "rf-pseudolabel-v1.0"
PSEUDO_CONTAM = 0.08       # rough prior anomaly rate for pseudo-labeling
PSEUDO_TOPFALLBACK = 0.05  # if IF yields one class, mark top 5% most anomalous as 1
THRESHOLD = 0.70           # threshold on RF probability for anomaly_flag

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # coerce numerics
    for col in ["bytes_in","bytes_out","packets_in","packets_out","duration_s"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).clip(lower=0)

    # protocol → integer code
    proto_map = {"TCP":0, "UDP":1, "ICMP":2}
    df["protocol_enc"] = df["protocol"].map(proto_map).fillna(3).astype(int)

    # scale for IsolationForest (helps when magnitudes differ a lot)
    scaler_if = StandardScaler()
    X_if = scaler_if.fit_transform(df[FEATURES])

    # RF can use raw features
    X_rf = df[FEATURES].values
    return df, X_if, X_rf

def iforest_anomaly_prob(decision_vals: np.ndarray):
    # decision_function: higher => more normal. Convert to anomaly-like 0..1
    s = (decision_vals - decision_vals.min()) / (decision_vals.max() - decision_vals.min() + 1e-9)
    return 1.0 - s

def make_pseudo_labels(X_if: np.ndarray):
    iforest = IsolationForest(n_estimators=200, contamination=PSEUDO_CONTAM, random_state=42)
    iforest.fit(X_if)
    dfun = iforest.decision_function(X_if)
    probs = iforest_anomaly_prob(dfun)

    # -1 = anomaly, 1 = normal
    y = (iforest.predict(X_if) == -1).astype(int)

    # ensure both classes exist
    if y.sum() == 0 or y.sum() == len(y):
        k = max(1, int(np.ceil(PSEUDO_TOPFALLBACK * len(y))))
        top_idx = np.argsort(-probs)[:k]
        y = np.zeros_like(y)
        y[top_idx] = 1
    return y

def fit_random_forest(X_rf: np.ndarray, y_pseudo: np.ndarray):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_rf, y_pseudo)
    return rf

def predict(rf: RandomForestClassifier, X_rf: np.ndarray, event_ids):
    proba = rf.predict_proba(X_rf)[:, 1]          # P(class=1 anomaly)
    flags = (proba >= THRESHOLD).astype(int)
    return [
        {"event_id": eid, "anomaly_score": float(p), "anomaly_flag": int(f),
         "model_version": MODEL_VERSION, "threshold": THRESHOLD}
        for eid, p, f in zip(event_ids, proba, flags)
    ]

if __name__ == "__main__":
    """
    Usage:
        python random_forest.py sample_events.csv > results.json

    Required input columns (no labels needed):
        event_id, bytes_in, bytes_out, packets_in, packets_out, duration_s, protocol
    """
    import sys
    df_raw = pd.read_csv(sys.argv[1])
    event_ids = df_raw["event_id"].astype(str).tolist()

    df, X_if, X_rf = preprocess(df_raw)
    y_pseudo = make_pseudo_labels(X_if)
    rf = fit_random_forest(X_rf, y_pseudo)
    results = predict(rf, X_rf, event_ids)

    print(json.dumps({"results": results, "model_version": MODEL_VERSION, "threshold": THRESHOLD}, indent=2))
