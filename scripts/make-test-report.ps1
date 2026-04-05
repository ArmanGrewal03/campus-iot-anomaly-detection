Param(
  [string]$GatewayUrl = "http://127.0.0.1:8003",
  [string[]]$Models = @("rf_latest","if_latest","ae_latest"),
  [string]$OutDir = "test_report",
  [switch]$OfflineOnly,
  [string]$VenvDir = ".report-venv",
  [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

# Resolve repo root (folder containing this script -> go up one)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent $scriptDir

$venvPath = Join-Path $repoRoot $VenvDir
$venvPython = Join-Path $venvPath "Scripts\python.exe"

function Get-SystemPython {
  $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonCmd) { return "python" }
  $pyCmd = Get-Command py -ErrorAction SilentlyContinue
  if ($pyCmd) { return "py -3" }
  throw "Python was not found on PATH. Install Python 3.x and rerun."
}

function Ensure-ReportVenv {
  param([string]$TargetVenvPath)
  if (-not (Test-Path $venvPython)) {
    Write-Host "Creating report virtual environment at '$TargetVenvPath'..."
    $systemPython = Get-SystemPython
    if ($systemPython -eq "py -3") {
      py -3 -m venv $TargetVenvPath
    } else {
      python -m venv $TargetVenvPath
    }
    if ($LASTEXITCODE -ne 0) {
      throw "Failed to create virtual environment at '$TargetVenvPath'."
    }
  }
}

function Ensure-Dependencies {
  if ($SkipInstall.IsPresent) {
    Write-Host "Skipping dependency installation (--SkipInstall)."
    return
  }

  # Quick import check; install only when needed. Be tolerant of failures.
  $importOk = $true
  try {
    & $venvPython -c "import numpy, matplotlib, seaborn, sklearn, requests" 1>$null 2>$null
    if ($LASTEXITCODE -ne 0) { $importOk = $false }
  } catch {
    $importOk = $false
  }
  if ($importOk) {
    Write-Host "Report dependencies already installed."
    return
  }

  Write-Host "Installing report dependencies (numpy, matplotlib, seaborn, scikit-learn, requests)..."
  & $venvPython -m pip install --upgrade pip
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip in report environment."
  }
  & $venvPython -m pip install numpy matplotlib seaborn scikit-learn requests
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to install report dependencies."
  }
}

Ensure-ReportVenv -TargetVenvPath $venvPath
Ensure-Dependencies

$argsList = @(
  "$scriptDir\generate_test_report.py",
  "--gateway-url", $GatewayUrl,
  "--out-dir", $OutDir
)

if ($OfflineOnly.IsPresent) {
  $argsList += @("--offline-only")
}

if ($Models.Count -gt 0) {
  $argsList += @("--models")
  $argsList += $Models
}

Write-Host "Generating test report..."
& $venvPython @argsList

if ($LASTEXITCODE -ne 0) {
  Write-Error "Report generation failed with exit code $LASTEXITCODE"
  exit $LASTEXITCODE
}

Write-Host "Report generated in '$OutDir'."
Write-Host "Open '$OutDir\report.md' for the summary and see PNGs for figures."

