# app.py
from flask import Flask, request, jsonify
from models import FlowRecord, BehaviorRecord
import storage
import traceback
import time

app = Flask(__name__)

# Simple model scoring placeholder
def score_record(feature_dict):
    """
    Placeholder for model inference.
    Replace with call to model server / local model prediction.
    Returns dict with 'anomaly_score' and 'predicted_label'
    """
    # very simple heuristic for demo:
    score = 0.0
    if feature_dict.get("flow_bytes_s") and feature_dict["flow_bytes_s"] > 1e6:
        score += 0.8
    if feature_dict.get("payload_entropy") and feature_dict["payload_entropy"] > 7.0:
        score += 0.5
    predicted = 1 if score > 0.6 else 0
    return {"anomaly_score": min(score, 1.0), "predicted_label": predicted}

@app.route("/ingest/flow", methods=["POST"])
def ingest_flow():
    """
    Accept a JSON payload for a single flow (or a list).
    """
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "empty payload"}), 400

    try:
        # batch or single
        if isinstance(payload, list):
            results = []
            for rec in payload:
                fr = FlowRecord(**rec)
                flat = fr.to_flat_dict()
                # store
                storage.store_flow_record(flat, rec)
                # score (synchronous demo)
                score = score_record(flat)
                results.append({"flow_id": flat.get("flow_id"), "score": score})
            return jsonify({"status": "ok", "results": results}), 201

        else:
            fr = FlowRecord(**payload)
            flat = fr.to_flat_dict()
            storage.store_flow_record(flat, payload)
            score = score_record(flat)
            return jsonify({"status": "ok", "flow_id": flat.get("flow_id"), "score": score}), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/ingest/behavior", methods=["POST"])
def ingest_behavior():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "empty payload"}), 400

    try:
        if isinstance(payload, list):
            results = []
            for rec in payload:
                br = BehaviorRecord(**rec)
                flat = br.to_flat_dict()
                storage.store_behavior_record(flat, rec)
                score = score_record(flat)  # reuse placeholder
                results.append({"device": flat.get("device_mac_hash"), "score": score})
            return jsonify({"status": "ok", "results": results}), 201
        else:
            br = BehaviorRecord(**payload)
            flat = br.to_flat_dict()
            storage.store_behavior_record(flat, payload)
            score = score_record(flat)
            return jsonify({"status": "ok", "device": flat.get("device_mac_hash"), "score": score}), 201
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": time.time()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
