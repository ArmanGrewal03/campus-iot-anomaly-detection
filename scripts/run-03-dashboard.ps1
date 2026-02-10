# PowerShell script to run the 03 Dashboard (React/Vite)
# This script installs dependencies and starts the React development server

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting 03 Dashboard (React/Vite)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Get the script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$serviceDir = Join-Path $projectRoot "03_Dashboard"

# Check if service directory exists
if (-not (Test-Path $serviceDir)) {
    Write-Host "Error: Service directory not found at $serviceDir" -ForegroundColor Red
    exit 1
}

# Change to service directory
Set-Location $serviceDir

# Check if npm is available
$npmCheck = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npmCheck) {
    Write-Host "Error: npm is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    
    # Check for winget (Windows Package Manager)
    $wingetCheck = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCheck) {
        Write-Host "Detected winget (Windows Package Manager). Would you like to install Node.js automatically?" -ForegroundColor Yellow
        Write-Host "You can run: winget install OpenJS.NodeJS.LTS" -ForegroundColor Cyan
        Write-Host ""
    }
    
    # Check for Chocolatey
    $chocoCheck = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCheck) {
        Write-Host "Detected Chocolatey. Would you like to install Node.js automatically?" -ForegroundColor Yellow
        Write-Host "You can run: choco install nodejs-lts -y" -ForegroundColor Cyan
        Write-Host ""
    }
    
    Write-Host "Manual installation:" -ForegroundColor Yellow
    Write-Host "1. Visit https://nodejs.org/" -ForegroundColor Cyan
    Write-Host "2. Download the LTS (Long Term Support) version for Windows" -ForegroundColor Cyan
    Write-Host "3. Run the installer and follow the setup wizard" -ForegroundColor Cyan
    Write-Host "4. Make sure to check 'Add to PATH' during installation" -ForegroundColor Cyan
    Write-Host "5. Restart your terminal/PowerShell after installation" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

# Check if node_modules exists
$nodeModulesPath = Join-Path $serviceDir "node_modules"
if (-not (Test-Path $nodeModulesPath)) {
    Write-Host "Node modules not found. Installing dependencies..." -ForegroundColor Yellow
    Write-Host "This may take a few minutes..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to install npm dependencies" -ForegroundColor Red
        exit 1
    }
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
}

# Start the React development server
Write-Host "Starting React development server..." -ForegroundColor Green
Write-Host "The dashboard will open automatically in your browser" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

npm run dev
