Param(
  [string]$Uri = "http://127.0.0.1:8003/test",
  [string]$DatasetName = "A",
  [string]$ModelName = "RFV1",
  [int]$Requests = 100,
  [string]$Body = "{}",
  [int]$TimeoutSec = 120
)

$ErrorActionPreference = "Stop"

if ($Requests -lt 1) {
  throw "Requests must be >= 1."
}

$headers = @{
  "Content-Type" = "application/json"
  "dataset_name" = $DatasetName
  "model_name"   = $ModelName
}

$latencies = New-Object System.Collections.Generic.List[Double]
$success = 0
$failed = 0

Write-Host "Benchmarking $Requests requests to $Uri"
Write-Host "dataset_name=$DatasetName model_name=$ModelName"

for ($i = 1; $i -le $Requests; $i++) {
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  try {
    Invoke-RestMethod -Method POST -Uri $Uri -Headers $headers -Body $Body -TimeoutSec $TimeoutSec | Out-Null
    $sw.Stop()
    $latencies.Add($sw.Elapsed.TotalMilliseconds)
    $success++
  } catch {
    $sw.Stop()
    $failed++
    Write-Warning "Request $i failed: $($_.Exception.Message)"
  }
}

if ($latencies.Count -eq 0) {
  throw "All requests failed. No latency percentiles can be computed."
}

$sorted = $latencies | Sort-Object

function Get-Percentile {
  Param(
    [double[]]$Values,
    [double]$Percent
  )
  $idx = [math]::Ceiling(($Percent / 100.0) * $Values.Count) - 1
  if ($idx -lt 0) { $idx = 0 }
  if ($idx -ge $Values.Count) { $idx = $Values.Count - 1 }
  return [math]::Round($Values[$idx], 2)
}

$p50 = Get-Percentile -Values $sorted -Percent 50
$p90 = Get-Percentile -Values $sorted -Percent 90
$p95 = Get-Percentile -Values $sorted -Percent 95
$min = [math]::Round(($sorted | Select-Object -First 1), 2)
$max = [math]::Round(($sorted | Select-Object -Last 1), 2)
$avg = [math]::Round((($sorted | Measure-Object -Average).Average), 2)

Write-Host ""
Write-Host "Results"
Write-Host "-------"
Write-Host "Total Requests : $Requests"
Write-Host "Successful     : $success"
Write-Host "Failed         : $failed"
Write-Host "Min            : $min ms"
Write-Host "Avg            : $avg ms"
Write-Host "P50            : $p50 ms"
Write-Host "P90            : $p90 ms"
Write-Host "P95            : $p95 ms"
Write-Host "Max            : $max ms"

