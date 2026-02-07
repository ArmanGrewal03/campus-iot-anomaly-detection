# Campus IoT Anomaly Detection - Startup Guide

## Consistent Ports
- **Backend API**: `http://localhost:8000`
- **Frontend Dashboard**: `http://localhost:5173`

## Prerequisites
1. Python virtual environment activated
2. Node.js installed

---

## Step 1: Start the Backend (Terminal 1)

```bash
# Navigate to backend directory
cd campus-iot-anomaly-detection/C-Backend/P1

# Activate Python virtual environment
source ../../venv/bin/activate

# Start FastAPI server
uvicorn main:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process...
```

---

## Step 2: Start the Frontend (Terminal 2)

```bash
# Navigate to frontend directory
cd campus-iot-anomaly-detection/D-Dashboard/V6

# Start Vite development server
npm run dev
```

**Expected Output:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

---

## Step 3: Access the Dashboard

Open your browser and navigate to:
```
http://localhost:5173/model
```

---

## Troubleshooting

### Port Already in Use Error

If you see `Error: Port 5173 is already in use`, kill the process:

```bash
# Find and kill the process using port 5173
lsof -i :5173
kill -9 <PID>

# Or for port 8000
lsof -i :8000
kill -9 <PID>
```

### API: Unreachable Error

1. **Check Backend is Running**: Visit `http://localhost:8000/api/health` in your browser.
   - Should return: `{"status":"healthy",...}`
   
2. **Hard Refresh Browser**: Press `Cmd + Shift + R` to clear cache.

3. **Check CORS**: Ensure port 5173 is in the backend's `allow_origins` list (already configured).

---

## Stopping the Services

Press `Ctrl + C` in each terminal window.
