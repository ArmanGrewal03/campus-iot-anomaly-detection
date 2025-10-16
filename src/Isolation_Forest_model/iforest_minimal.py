# ml/iforest_minimal.py
import json, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

FEATURES = ["bytes_in","bytes_out","packets_in","packets_out","duration_s","protocol_enc"]
MODEL_VERSION = "iforest-v1.0"
THRESHOLD = 0.70  # adjust after quick eval

def preprocess(df: pd.DataFrame):
    df = df.copy()
    # basic clean
    for col in ["bytes_in","bytes_out","packets_in","packets_out","duration_s"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).clip(lower=0)
    # simple protocol encoding (replace with your map/one-hot)
    proto_map = {"TCP":0, "UDP":1, "ICMP":2}
    df["protocol_enc"] = df["protocol"].map(proto_map).fillna(3).astype(int)
    # scale numeric
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[FEATURES])
    return X, scaler

def fit_iforest(X):
    # contamination is rough prior of anomalies; tune later
    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    clf.fit(X)
    return clf

def score_to_prob(scores):
    # sklearn gives anomaly score via decision_function; convert to 0..1
    # decision_function: higher = more normal; invert it
    s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return 1.0 - s  # higher = more anomalous

def predict(clf, X, event_ids):
    # decision_function returns "normality" score
    dfun = clf.decision_function(X)
    probs = score_to_prob(dfun)
    flags = (probs >= THRESHOLD).astype(int)
    return [
        {"event_id": eid, "anomaly_score": float(p), "anomaly_flag": int(f),
         "model_version": MODEL_VERSION, "threshold": THRESHOLD}
        for eid, p, f in zip(event_ids, probs, flags)
    ]

if __name__ == "__main__":
    # Example usage: python iforest_minimal.py sample.csv > results.json
    import sys
    df = pd.read_csv(sys.argv[1])
    event_ids = df["event_id"].astype(str).tolist()
    X, scaler = preprocess(df)
    clf = fit_iforest(X)               # for week 7 demo: fit on same data
    results = predict(clf, X, event_ids)
    print(json.dumps({"results": results, "model_version": MODEL_VERSION, "threshold": THRESHOLD}, indent=2))
