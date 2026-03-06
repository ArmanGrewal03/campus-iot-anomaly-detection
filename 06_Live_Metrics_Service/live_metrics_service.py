"""
Live Metrics Service — generates mock time-series data for dashboard tiles.
X-axis = time (last 5 minutes, one point every 5 seconds).
Y-axis = meaningful metrics; series have coherent trends and realistic spikes.
Do not modify existing services; this is a standalone service.
"""
import random
import os
import math
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Live Metrics Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Last 2 minutes, one sample every 2 seconds = 60 points — faster shift
POINTS = 60
INTERVAL_SECONDS = 2
WINDOW_SECONDS = POINTS * INTERVAL_SECONDS  # 120 = 2 min


def _timestamp_labels(now_ts: int) -> List[str]:
    """X-axis: actual timestamps for each 5s slot so the graph shifts with time (e.g. 10:31:05)."""
    labels: List[str] = []
    for i in range(POINTS):
        slot_ts = now_ts - (POINTS - 1 - i) * INTERVAL_SECONDS
        dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        labels.append(dt.strftime("%H:%M:%S"))
    return labels


def _slot_ts_list(now_ts: int) -> List[int]:
    """Return the wall-clock slot timestamp for each of the 60 points (oldest to newest)."""
    return [now_ts - (POINTS - 1 - i) * INTERVAL_SECONDS for i in range(POINTS)]


def _slot_value(slot_ts: int, series_id: int, base: float, wave_amp: float,
                wave_period_sec: float, spike_period_sec: float, spike_height: float,
                spike_width_sec: float, min_val: float, max_val: float,
                phase: float = 0) -> float:
    """
    Compute value for a single time slot based purely on absolute time.
    Every parameter is in seconds so features shift as the window slides.
    """
    # Wave: smooth oscillation tied to absolute time
    v = base + wave_amp * math.sin(2 * math.pi * slot_ts / wave_period_sec + phase)
    # Spike: repeats every spike_period_sec; spike_width_sec wide
    spike_phase = (slot_ts % spike_period_sec) / spike_period_sec
    spike_frac = spike_width_sec / spike_period_sec
    if spike_phase < spike_frac:
        # Triangular spike: ramps up then down
        half = spike_frac / 2
        if spike_phase < half:
            v += spike_height * (spike_phase / half)
        else:
            v += spike_height * (1 - (spike_phase - half) / half)
    # Small deterministic jitter seeded on (slot_ts, series_id) so it's stable
    rng = random.Random(slot_ts * 31 + series_id * 7)
    v += (rng.random() - 0.5) * 5
    return max(min_val, min(max_val, round(v, 1)))


# Shared timing so all tiles are synchronized (same wave cycle, same spike cycle)
_WAVE_PERIOD = 60      # seconds — one full wave cycle per 60s
_SPIKE_PERIOD = 80     # seconds — spike repeats every 80s
_SPIKE_WIDTH = 16      # seconds — how wide each spike is

def _request_status_series(slot_ts_list: List[int], _seed: int) -> List[float]:
    """Requests: synchronized wave + spikes, highest amplitude."""
    return [
        _slot_value(ts, series_id=1, base=80, wave_amp=25,
                    wave_period_sec=_WAVE_PERIOD, spike_period_sec=_SPIKE_PERIOD,
                    spike_height=60, spike_width_sec=_SPIKE_WIDTH,
                    min_val=20, max_val=220, phase=0)
        for ts in slot_ts_list
    ]


def _blocking_status_series(slot_ts_list: List[int], _seed: int) -> List[float]:
    """Blocking: same timing, lower amplitude — moves in sync with requests."""
    return [
        _slot_value(ts, series_id=2, base=30, wave_amp=10,
                    wave_period_sec=_WAVE_PERIOD, spike_period_sec=_SPIKE_PERIOD,
                    spike_height=30, spike_width_sec=_SPIKE_WIDTH,
                    min_val=0, max_val=95, phase=0.3)
        for ts in slot_ts_list
    ]


def _query_per_second_series(slot_ts_list: List[int], _seed: int) -> List[float]:
    """QPS: deterministic bursts at same spike timing — synchronized with other tiles."""
    out: List[float] = []
    for ts in slot_ts_list:
        rng = random.Random(ts * 13 + 99)
        spike_phase = (ts % _SPIKE_PERIOD) / _SPIKE_PERIOD
        spike_frac = _SPIKE_WIDTH / _SPIKE_PERIOD
        if spike_phase < spike_frac:
            out.append(round(rng.uniform(15, 72), 1))
        elif spike_phase < spike_frac * 1.5:
            out.append(round(rng.uniform(4, 20), 1))
        else:
            out.append(0.0)
    return out


def _packet_rate_series(slot_ts_list: List[int], _seed: int) -> List[float]:
    """Packets: same timing, medium amplitude — synchronized with other tiles."""
    return [
        _slot_value(ts, series_id=4, base=65, wave_amp=20,
                    wave_period_sec=_WAVE_PERIOD, spike_period_sec=_SPIKE_PERIOD,
                    spike_height=45, spike_width_sec=_SPIKE_WIDTH,
                    min_val=10, max_val=180, phase=0.15)
        for ts in slot_ts_list
    ]


@app.get("/health")
async def health():
    return JSONResponse(
        content={
            "status": "ok",
            "service": "live-metrics",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        status_code=200,
    )


@app.get("/metrics")
async def get_metrics():
    """
    Returns time-series for dashboard tiles.
    X-axis: timestamps (HH:MM:SS) for each 5s slot so the graph shifts with time.
    Values are deterministic per slot so overlapping windows match and the chart moves smoothly.
    """
    now_ts = int(datetime.now(timezone.utc).timestamp())
    # Snap to nearest 5s boundary so the graph updates cleanly on each slot
    now_ts = (now_ts // INTERVAL_SECONDS) * INTERVAL_SECONDS
    slot_ts_list = _slot_ts_list(now_ts)
    labels = _timestamp_labels(now_ts)

    request_status = _request_status_series(slot_ts_list, 0)
    blocking_status = _blocking_status_series(slot_ts_list, 0)
    query_per_second = _query_per_second_series(slot_ts_list, 0)
    packet_rate = _packet_rate_series(slot_ts_list, 0)

    return JSONResponse(
        content={
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "labels": labels,
            "x_axis_description": "Time (last 5 minutes)",
            "y_axis_request_status": "Requests (per 5s interval)",
            "y_axis_blocking_status": "Blocked events (per 5s interval)",
            "y_axis_query_per_second": "Queries per interval",
            "packet_rate": packet_rate,
            "request_status": request_status,
            "blocking_status": blocking_status,
            "query_per_second": query_per_second,
            "max_request_status": max(request_status) if request_status else 0,
            "max_blocking_status": max(blocking_status) if blocking_status else 0,
        },
        status_code=200,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("LIVE_METRICS_PORT", "8010"))
    uvicorn.run(app, host="127.0.0.1", port=port)
