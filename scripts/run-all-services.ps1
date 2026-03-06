# PowerShell script to run all services simultaneously
# This script starts the Data Ingestion Service, Model Service, User Service, and Dashboard in separate windows
# It also opens the website and provides a way to terminate all services
#
# To run each service in a separate terminal TAB inside Cursor: use the built-in task instead:
#   Ctrl+Shift+P (Command Palette) -> "Tasks: Run Task" -> "Run All Services (Cursor terminals)"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting All Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Paths to individual scripts
$script01 = Join-Path $scriptDir "run-01-data-ingestion.ps1"
$script02 = Join-Path $scriptDir "run-02-model-service.ps1"
$script04 = Join-Path $scriptDir "run-04-user-service.ps1"
$script03 = Join-Path $scriptDir "run-03-dashboard.ps1"
$script05 = Join-Path $scriptDir "run-05-gateway.ps1"
$script06 = Join-Path $scriptDir "run-06-live-metrics.ps1"

# Check if scripts exist
if (-not (Test-Path $script01)) {
    Write-Host "Error: Script not found: $script01" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $script02)) {
    Write-Host "Error: Script not found: $script02" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $script04)) {
    Write-Host "Error: Script not found: $script04" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $script03)) {
    Write-Host "Error: Script not found: $script03" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $script05)) {
    Write-Host "Warning: Gateway script not found: $script05 (optional)" -ForegroundColor Yellow
}
if (-not (Test-Path $script06)) {
    Write-Host "Warning: Live Metrics script not found: $script06 (optional)" -ForegroundColor Yellow
}

Write-Host "Starting services in separate PowerShell windows..." -ForegroundColor Green
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Cyan
Write-Host "  - Data Ingestion API: http://127.0.0.1:8000" -ForegroundColor White
Write-Host "  - Model API:          http://127.0.0.1:8001" -ForegroundColor White
Write-Host "  - User Service:       http://127.0.0.1:8002 (WebSocket: ws://127.0.0.1:8002/ws/data-stream)" -ForegroundColor White
Write-Host "  - API Gateway:        http://127.0.0.1:8003 (optional)" -ForegroundColor White
Write-Host "  - Live Metrics:       http://127.0.0.1:8010 (optional)" -ForegroundColor White
Write-Host "  - Dashboard:          http://127.0.0.1:5173 (will open automatically)" -ForegroundColor White
Write-Host ""

# Store process IDs for later termination
$processIds = @()

# Start each service in a new PowerShell window and capture process IDs
Write-Host "Starting Data Ingestion Service..." -ForegroundColor Yellow
$proc01 = Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script01`"" -PassThru
$processIds += $proc01.Id
Start-Sleep -Seconds 2

Write-Host "Starting Model Service..." -ForegroundColor Yellow
$proc02 = Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script02`"" -PassThru
$processIds += $proc02.Id
Start-Sleep -Seconds 2

Write-Host "Starting User Service..." -ForegroundColor Yellow
$proc04 = Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script04`"" -PassThru
$processIds += $proc04.Id
Start-Sleep -Seconds 2

Write-Host "Starting Dashboard..." -ForegroundColor Yellow
$proc03 = Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script03`"" -PassThru
$processIds += $proc03.Id
Start-Sleep -Seconds 2

# Start Gateway if script exists (optional)
if (Test-Path $script05) {
    Write-Host "Starting API Gateway..." -ForegroundColor Yellow
    $proc05 = Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script05`"" -PassThru
    $processIds += $proc05.Id
    Start-Sleep -Seconds 2
}

# Start Live Metrics Service if script exists (optional)
if (Test-Path $script06) {
    Write-Host "Starting Live Metrics Service..." -ForegroundColor Yellow
    $proc06 = Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script06`"" -PassThru
    $processIds += $proc06.Id
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "All services are starting..." -ForegroundColor Green
Write-Host "Waiting for services to initialize..." -ForegroundColor Yellow

# Wait a bit longer for the dashboard to start
Start-Sleep -Seconds 8

# Open the website in the default browser
$dashboardUrl = "http://localhost:5173/Home"
Write-Host ""
Write-Host "Opening dashboard in browser: $dashboardUrl" -ForegroundColor Cyan
try {
    Start-Process $dashboardUrl
    Write-Host "Browser opened successfully!" -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not open browser automatically. Please navigate to $dashboardUrl" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Services are running!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Process IDs:" -ForegroundColor Cyan
Write-Host "  - Data Ingestion Service: PID $($proc01.Id)" -ForegroundColor White
Write-Host "  - Model Service:          PID $($proc02.Id)" -ForegroundColor White
Write-Host "  - User Service:           PID $($proc04.Id)" -ForegroundColor White
Write-Host "  - Dashboard:              PID $($proc03.Id)" -ForegroundColor White
if (Test-Path $script05) { Write-Host "  - API Gateway:             PID $($proc05.Id)" -ForegroundColor White }
if (Test-Path $script06) { Write-Host "  - Live Metrics Service:    PID $($proc06.Id)" -ForegroundColor White }
Write-Host ""
Write-Host "Press 'Q' and Enter to terminate all services, or any other key to exit (services will continue running)..." -ForegroundColor Yellow

# Function to terminate all services
function Stop-AllServices {
    Write-Host ""
    Write-Host "Terminating all services..." -ForegroundColor Yellow
    
    foreach ($pid in $processIds) {
        try {
            $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "Stopping process PID $pid..." -ForegroundColor Yellow
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            }
        } catch {
            Write-Host "Could not stop process PID $pid (may have already terminated)" -ForegroundColor Gray
        }
    }
    
    # Also try to kill processes by port (in case PIDs changed)
    Write-Host "Cleaning up processes using service ports..." -ForegroundColor Yellow
    
    # Function to kill process using a specific port
    function Stop-ProcessByPort {
        param([int]$Port)
        try {
            $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
            if ($connection) {
                $pid = $connection.OwningProcess
                if ($pid) {
                    Write-Host "Stopping process on port $Port (PID $pid)..." -ForegroundColor Yellow
                    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                }
            }
        } catch {
            # Ignore errors (port might not be in use)
        }
    }
    
    # Stop processes on known service ports
    Stop-ProcessByPort -Port 8000  # Data Ingestion Service
    Stop-ProcessByPort -Port 8001  # Model Service
    Stop-ProcessByPort -Port 8002  # User Service
    Stop-ProcessByPort -Port 8003  # API Gateway
    Stop-ProcessByPort -Port 8010  # Live Metrics Service
    Stop-ProcessByPort -Port 5173  # Dashboard
    
    Write-Host ""
    Write-Host "All services terminated!" -ForegroundColor Green
    Write-Host "Press any key to exit..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# Wait for user input
$userInput = Read-Host
if ($userInput -eq "Q" -or $userInput -eq "q") {
    Stop-AllServices
} else {
    Write-Host ""
    Write-Host "Services will continue running in their respective windows." -ForegroundColor Yellow
    Write-Host "Close the individual windows to stop each service." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
