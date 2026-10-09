<#
Start the demo: the backend (FastAPI, port 8000) and the dashboard (Vite, port 5173), hidden in the
background with their logs in results\, then print the URL. No Docker needed: the cameras are the
clips in data\india_dataset\raw_videos, read by the backend itself (FRAME_SOURCE=local).

    powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1                  # alerts as set in .env
    powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -ConsoleAlerts   # no Telegram messages
    powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -InMemory        # do not write to MongoDB
    powershell -ExecutionPolicy Bypass -File scripts\stop_demo.ps1

Running it again restarts the backend it started before. A dashboard already running on port 5173
(e.g. `npm run dev` in a terminal) is reused.
#>
param(
    [switch]$ConsoleAlerts,
    [switch]$InMemory,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "venv\Scripts\python.exe"
$results = Join-Path $root "results"
$pidFile = Join-Path $results "demo.pids"

if (-not (Test-Path $python)) { throw "No virtual environment at $python (see README: python -m venv venv)." }
if (-not (Test-Path (Join-Path $root "frontend\node_modules"))) { throw "Run 'npm install' in frontend first." }
$clips = Get-ChildItem (Join-Path $root "data\india_dataset\raw_videos") -Include *.mp4, *.avi, *.mov, *.mkv -Recurse -ErrorAction SilentlyContinue
if (-not $clips) { Write-Warning "No clips in data\india_dataset\raw_videos: the camera zones will show 'No signal'." }

function Listener([int]$port) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
}

function Wait-Http([string]$url, [int]$seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try { return Invoke-RestMethod $url -TimeoutSec 5 } catch { Start-Sleep -Seconds 2 }
    }
    return $null
}

# Stop the backend (and dashboard) this script started last time.
if (Test-Path $pidFile) {
    foreach ($id in Get-Content $pidFile) { & taskkill /PID $id /T /F 2>$null | Out-Null }
    Remove-Item $pidFile
    Start-Sleep -Seconds 2
}
$busy = Listener $BackendPort
if ($busy) {
    throw "Port $BackendPort is in use by process $($busy.OwningProcess). Stop it (scripts\stop_demo.ps1, or close that terminal) and run this again."
}

# Backend settings: everything else comes from .env.
$env:FRAME_SOURCE = "local"
if ($ConsoleAlerts) { $env:ALERT_CHANNEL = "console" }
if ($InMemory) { $env:MONGODB_URI = " " }  # a blank value would delete the variable and .env's URI would be used

$started = @()
$backend = Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$BackendPort" `
    -WorkingDirectory (Join-Path $root "backend") -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $results "demo_backend.out.log") -RedirectStandardError (Join-Path $results "demo_backend.log")
$started += $backend.Id

if (Listener $FrontendPort) {
    Write-Host "Dashboard: reusing the server already on port $FrontendPort."
} else {
    $frontend = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev", "--", "--port", "$FrontendPort", "--strictPort" `
        -WorkingDirectory (Join-Path $root "frontend") -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $results "demo_frontend.log") -RedirectStandardError (Join-Path $results "demo_frontend.err.log")
    $started += $frontend.Id
}
$started | Set-Content $pidFile

Write-Host "Starting the backend (loading CSRNet and YOLO takes up to a minute)..."
$health = Wait-Http "http://localhost:$BackendPort/health" 180
if (-not $health) { throw "The backend did not start; see results\demo_backend.log" }
if (-not (Wait-Http "http://localhost:$FrontendPort/" 90)) { throw "The dashboard did not start; see results\demo_frontend.log" }

$detector = if ($health.detector) { "$($health.detector.model) on $($health.detector.device)" } else { "off ($($health.detector_error))" }
Write-Host ""
Write-Host "Crowd model : $(if ($health.model_loaded) { $health.model.label } else { $health.model_error })"
Write-Host "Detector    : $detector"
Write-Host "Frames      : $($health.frame_source), zones $($health.zones -join ', ')"
Write-Host "Storage     : $($health.storage)$(if ($ConsoleAlerts) { ', alerts to the console only' })"
Write-Host ""
Write-Host "Dashboard   : http://localhost:$FrontendPort" -ForegroundColor Green
Write-Host "API docs    : http://localhost:$BackendPort/docs"
Write-Host "Logs        : results\demo_backend.log, results\demo_frontend.log   Stop: scripts\stop_demo.ps1"
