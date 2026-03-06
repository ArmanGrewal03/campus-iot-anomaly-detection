# 06 Live Metrics Service

Standalone service that generates **mock live time-series data** for dashboard tiles. It does not modify any existing services.

## Purpose

- Provides data for "live" line and bar charts on the Home page (Requests Status, Blocking Status, Query Per Second).
- Metrics are relevant to campus IoT / network anomaly detection: packet rate, request activity, blocking events, QPS-style bursts.
- Data is designed to **spike and vary** so charts look dynamic and live.

## Endpoints

- **GET /health** — Health check.
- **GET /metrics** — Returns JSON with:
  - `packet_rate` — Time-series (60 points), spiky.
  - `request_status` — Request/activity series, smooth spikes; used by "Requests Status" tile.
  - `blocking_status` — Blocked events series; used by "Blocking Status" tile.
  - `query_per_second` — QPS-style bursts (many zeros, occasional spikes); used by "Query Per Second" tile.
  - `max_request_status`, `max_blocking_status` — Current max values for display.

## Run

```bash
# From project root
.\scripts\run-06-live-metrics.ps1
```

Or manually:

```bash
cd 06_Live_Metrics_Service
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn live_metrics_service:app --host 127.0.0.1 --port 8010 --reload
```

Service runs at **http://127.0.0.1:8010**. The dashboard polls `/metrics` every 3 seconds when the Live tiles are visible.

## Port

Default: **8010**. Override with:

```bash
set LIVE_METRICS_PORT=8012
python -m uvicorn live_metrics_service:app --host 127.0.0.1 --port 8012 --reload
```

If you change the port, update `LIVE_METRICS_BASE` in the dashboard components: `LiveRequestsStatusTile.tsx`, `LiveBlockingStatusTile.tsx`, `LiveQueryPerSecondTile.tsx`.
