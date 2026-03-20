# PowerShell script to run the 07 LLM Service (LangChain + Groq)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting 07 LLM Service" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$serviceDir = Join-Path $projectRoot "07_LLM_Service"

if (-not (Test-Path $serviceDir)) {
    Write-Host "Error: Service directory not found at $serviceDir" -ForegroundColor Red
    exit 1
}

Set-Location $serviceDir

$venvPath = Join-Path $serviceDir "venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPath) -or -not (Test-Path $venvPython)) {
    if (Test-Path $venvPath) { Remove-Item -Recurse -Force $venvPath }
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

& "$venvPath\Scripts\Activate.ps1"

Write-Host "Installing/updating requirements..." -ForegroundColor Yellow
python -m pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install dependencies" -ForegroundColor Red
    exit 1
}

if (-not $env:GROQ_API_KEY) {
    Write-Host "Warning: GROQ_API_KEY is not set. /llm/chat will fail until this env var is configured." -ForegroundColor Yellow
}

Write-Host "Starting LLM Service on http://127.0.0.1:8004" -ForegroundColor Green
Write-Host "Endpoints: GET /health, POST /llm/chat" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

python -m uvicorn llm_service:app --host 127.0.0.1 --port 8004 --reload
