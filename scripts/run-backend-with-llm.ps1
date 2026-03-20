# PowerShell script to run backend services + gateway + LLM service (no dashboard)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Backend + Gateway + LLM" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not $env:GROQ_API_KEY) {
    Write-Host "Warning: GROQ_API_KEY is not set. LLM chat calls will fail until it is configured." -ForegroundColor Yellow
    Write-Host "Set it with: `$env:GROQ_API_KEY=`"your_key`"" -ForegroundColor Yellow
    Write-Host ""
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$script01 = Join-Path $scriptDir "run-01-data-ingestion.ps1"
$script02 = Join-Path $scriptDir "run-02-model-service.ps1"
$script04 = Join-Path $scriptDir "run-04-user-service.ps1"
$script05 = Join-Path $scriptDir "run-05-gateway.ps1"
$script07 = Join-Path $scriptDir "run-07-llm-service.ps1"

$requiredScripts = @($script01, $script02, $script04, $script05, $script07)
foreach ($script in $requiredScripts) {
    if (-not (Test-Path $script)) {
        Write-Host "Error: Required script not found: $script" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Starting services in separate PowerShell windows..." -ForegroundColor Green
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Cyan
Write-Host "  - Data Ingestion API: http://127.0.0.1:8000" -ForegroundColor White
Write-Host "  - Model API:          http://127.0.0.1:8001" -ForegroundColor White
Write-Host "  - User Service:       http://127.0.0.1:8002" -ForegroundColor White
Write-Host "  - API Gateway:        http://127.0.0.1:8003" -ForegroundColor White
Write-Host "  - LLM Service:        http://127.0.0.1:8004" -ForegroundColor White
Write-Host ""

$processIds = @()

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

Write-Host "Starting LLM Service..." -ForegroundColor Yellow
$proc07 = Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script07`"" -PassThru
$processIds += $proc07.Id
Start-Sleep -Seconds 2

Write-Host "Starting API Gateway..." -ForegroundColor Yellow
$proc05 = Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$script05`"" -PassThru
$processIds += $proc05.Id
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "Services are starting..." -ForegroundColor Green
Write-Host "Press 'Q' and Enter to terminate all started services, or any other key to exit." -ForegroundColor Yellow

function Stop-AllServices {
    Write-Host ""
    Write-Host "Terminating started services..." -ForegroundColor Yellow

    foreach ($pid in $processIds) {
        try {
            $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "Stopping process PID $pid..." -ForegroundColor Yellow
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            }
        } catch {
            Write-Host "Could not stop process PID $pid (may have already terminated)." -ForegroundColor Gray
        }
    }

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
            # Ignore port cleanup errors
        }
    }

    Stop-ProcessByPort -Port 8000
    Stop-ProcessByPort -Port 8001
    Stop-ProcessByPort -Port 8002
    Stop-ProcessByPort -Port 8003
    Stop-ProcessByPort -Port 8004

    Write-Host ""
    Write-Host "All started services terminated." -ForegroundColor Green
}

$userInput = Read-Host
if ($userInput -eq "Q" -or $userInput -eq "q") {
    Stop-AllServices
} else {
    Write-Host ""
    Write-Host "Services will continue running in their windows." -ForegroundColor Yellow
}
