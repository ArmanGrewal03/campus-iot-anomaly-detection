# Service Startup Scripts

This directory contains PowerShell scripts to run the individual services and all services together.

## Available Scripts

### Individual Service Scripts

1. **run-01-data-ingestion.ps1**
   - Starts the Data Ingestion Service (FastAPI)
   - Runs on: http://127.0.0.1:8000
   - Automatically activates virtual environment and installs dependencies

2. **run-02-model-service.ps1**
   - Starts the Model Service (FastAPI)
   - Runs on: http://127.0.0.1:8001
   - Automatically activates virtual environment and installs dependencies

3. **run-04-user-service.ps1**
   - Starts the User Service (FastAPI with WebSocket)
   - Runs on: http://127.0.0.1:8002
   - WebSocket endpoint: ws://127.0.0.1:8002/ws/data-stream
   - Automatically activates virtual environment and installs dependencies

4. **run-03-dashboard.ps1**
   - Starts the Vue.js Dashboard
   - Runs on: http://127.0.0.1:8080 (default Vue CLI port)
   - Automatically installs npm dependencies if needed

### Combined Script

5. **run-all-services.ps1**
   - Starts all three services simultaneously
   - Each service runs in its own PowerShell window
   - Useful for development when you need all services running

## Usage

### Run Individual Service

```powershell
# From the project root
.\scripts\run-01-data-ingestion.ps1
.\scripts\run-02-model-service.ps1
.\scripts\run-04-user-service.ps1
.\scripts\run-03-dashboard.ps1
```

### Run All Services

```powershell
# From the project root
.\scripts\run-all-services.ps1
```

This will open four separate PowerShell windows, one for each service.

## Prerequisites

- **Python 3.x** - Required for services 01, 02, and 04
- **Node.js and npm** - Required for service 03 (Dashboard)
- **PowerShell** - Required to run the scripts (Windows default)

## Service Ports

- **Data Ingestion Service**: Port 8000
- **Model Service**: Port 8001
- **User Service**: Port 8002 (WebSocket: ws://127.0.0.1:8002/ws/data-stream)
- **Dashboard**: Port 8080 (default Vue CLI dev server)

## Stopping Services

- **Individual scripts**: Press `Ctrl+C` in the service window
- **All services script**: Close each individual service window

## Notes

- The scripts automatically create virtual environments if they don't exist
- Dependencies are automatically installed/updated when services start
- The Dashboard script will open your default browser automatically
- All services use `--reload` flag for development (auto-reload on code changes)
