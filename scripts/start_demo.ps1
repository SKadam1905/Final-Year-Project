<#
Start the demo: the backend (FastAPI, port 8000) and the dashboard (Vite, port 5173), hidden in the
background with their logs in results\, then print the URL. No Docker needed: the cameras are the
clips in data\india_dataset\raw_videos, read by the backend itself (FRAME_SOURCE=local).

    powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1                  # alerts as set in .env
    powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -ConsoleAlerts   # no Telegram messages
    powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -InMemory        # do not write to MongoDB
    powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -Force           # stop whatever holds the ports
    powershell -ExecutionPolicy Bypass -File scripts\stop_demo.ps1

Running it again restarts the backend it started before. A dashboard of this project already running
on port 5173 (e.g. `npm run dev` in a terminal) is reused. If another program holds port 8000 or
5173, the script asks before stopping it (-Force: without asking), or prints the command to do it.
Without internet the demo still runs: MongoDB falls back to in-memory storage and Telegram alerts
fail (shown on the dashboard); the script says so.
#>
param(
    [switch]$ConsoleAlerts,
    [switch]$InMemory,
    [switch]$Force,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "venv\Scripts\python.exe"
$results = Join-Path $root "results"
$pidFile = Join-Path $results "demo.pids"

function Fail([string]$message) {
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $python)) { Fail "No virtual environment at $python (see README: python -m venv venv)." }
if (-not (Test-Path (Join-Path $root "frontend\node_modules"))) { Fail "Run 'npm install' in frontend first." }
$clips = Get-ChildItem (Join-Path $root "data\india_dataset\raw_videos") -Include *.mp4, *.avi, *.mov, *.mkv -Recurse -ErrorAction SilentlyContinue
if (-not $clips) { Write-Warning "No clips in data\india_dataset\raw_videos: the camera zones will show 'No signal'." }

# taskkill through cmd: a process that is already gone (e.g. a PID file left from before a reboot) is not
# an error. Called directly, PowerShell 5.1 turns taskkill's message into an error that stops the script.
function Stop-Tree([string]$id) {
    & cmd /c "taskkill /PID $id /T /F >nul 2>&1"
    return $LASTEXITCODE -eq 0
}

# A PID in the file is only ours if its command line says so: after a reboot Windows may have given
# that number to another program, which must not be killed.
function Test-Ours([string]$id) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$id" -ErrorAction SilentlyContinue
    return $proc -and ($proc.CommandLine -match 'uvicorn.*app\.main:app' -or $proc.CommandLine -match 'npm.*run.*dev' -or
                       $proc.CommandLine -like "*nashik-kumbh-stampede-predictor*vite*")
}

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

function Test-Reachable([string]$hostName, [int]$port, [int]$timeoutMs = 3000) {
    $client = New-Object System.Net.Sockets.TcpClient
    try { return $client.ConnectAsync($hostName, $port).Wait($timeoutMs) -and $client.Connected }
    catch { return $false }  # DNS failure (offline) or refused
    finally { $client.Close() }
}

# Settings as the backend will see them: the environment first, then .env (values never printed).
$envFile = Join-Path $root ".env"
$dotenv = @{}
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') { $dotenv[$Matches[1]] = $Matches[2].Trim('"', "'") }
    }
} else {
    Write-Warning ".env not found: running on defaults. Copy .env.example to .env to set the model, MongoDB and Telegram."
}
function Setting([string[]]$names, [string]$default = "") {
    foreach ($name in $names) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value -and $value.Trim()) { return $value.Trim() }
        if ($dotenv[$name]) { return $dotenv[$name] }
    }
    return $default
}

# Free the ports: our own previous run first, then ask about anything else.
if (Test-Path $pidFile) {
    foreach ($id in Get-Content $pidFile) { if (Test-Ours $id) { Stop-Tree $id | Out-Null } }
    Remove-Item $pidFile
    Start-Sleep -Seconds 2
}

function Free-Port([int]$port, [string]$what) {
    $listener = Listener $port
    if (-not $listener) { return }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    $who = "$($proc.Name) (pid $($proc.ProcessId))"
    $command = "taskkill /PID $($proc.ProcessId) /T /F"
    $canAsk = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected
    $stop = $Force -or ($canAsk -and ((Read-Host "Port $port ($what) is in use by $who. Stop it? [y/N]") -match '^(y|yes)$'))
    if (-not $stop) {
        Fail "Port $port ($what) is in use by $who. Stop it with:`n    $command`nor start this script with -Force to stop it."
    }
    Stop-Tree $proc.ProcessId | Out-Null
    Start-Sleep -Seconds 2
    if (Listener $port) { Fail "Could not free port $port; stop $who yourself:`n    $command" }
    Write-Host "Stopped $who, which held port $port."
}

Free-Port $BackendPort "backend"
$reuseDashboard = $false
$dash = Listener $FrontendPort
if ($dash) {
    $dashCmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($dash.OwningProcess)").CommandLine
    if ($dashCmd -like "*nashik-kumbh-stampede-predictor*vite*") { $reuseDashboard = $true }
    else { Free-Port $FrontendPort "dashboard" }
}

# Backend settings: everything else comes from .env.
$env:FRAME_SOURCE = "local"
if ($ConsoleAlerts) { $env:ALERT_CHANNEL = "console" }
if ($InMemory) { $env:MONGODB_URI = " " }  # a blank value would delete the variable and .env's URI would be used

$weights = Setting "CSRNET_WEIGHTS", "MODEL_PATH" "backend/ml/weights/csrnet_best.pth"
if (-not (Test-Path (Join-Path $root $weights))) {
    Write-Warning "CSRNET_WEIGHTS ($weights) not found: the dashboard will say 'No trained model: counts unavailable'. Set CSRNET_WEIGHTS=backend/ml/weights/public/csrnet_partA_public.pth in .env."
}
$yolo = Setting "YOLO_WEIGHTS" "backend/ml/weights/yolo/yolov8s.pt"
if ($yolo -notin "off", "none", "0", "false" -and -not (Test-Path (Join-Path $root $yolo))) {
    Write-Warning "YOLO_WEIGHTS ($yolo) not found: counts will come from CSRNet only. Download: cd backend; ..\venv\Scripts\python.exe -m tools.get_yolo_weights yolov8s"
}
$channels = (Setting "ALERT_CHANNEL" "console").ToLower()
$telegramNote = "off"
if ($channels -match "telegram|all") {
    if (-not (Setting "TELEGRAM_BOT_TOKEN") -or -not (Setting "TELEGRAM_CHAT_ID")) {
        $telegramNote = "NOT SET UP (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in .env): alerts on the dashboard and console only"
        Write-Warning "ALERT_CHANNEL includes telegram but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is empty in .env."
    } elseif (Test-Reachable "api.telegram.org" 443) {
        $telegramNote = "on"
    } else {
        $telegramNote = "UNREACHABLE (no internet?): alerts will show on the dashboard, marked as not delivered"
        Write-Warning "api.telegram.org cannot be reached: Telegram alerts will fail; the dashboard still shows them."
    }
}
$mongoWanted = [bool](Setting "MONGODB_URI", "MONGO_URI", "MONGODB_URL")
$usersFile = Join-Path $root (Setting "USERS_CONFIG" "config/users.json")
$accounts = @()
if (Test-Path $usersFile) { $accounts = @((Get-Content $usersFile -Raw | ConvertFrom-Json).users) }
if (-not $accounts.Count) {
    Write-Warning "No dashboard accounts yet: create one with  cd backend; ..env\Scripts\python.exe -m tools.create_user admin --role admin --generate"
}
if (-not (Setting "SECRET_KEY")) {
    Write-Warning "SECRET_KEY is not set in .env: everyone is logged out when the backend restarts. tools.create_user adds one."
}

$started = @()
$backend = Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$BackendPort" `
    -WorkingDirectory (Join-Path $root "backend") -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $results "demo_backend.out.log") -RedirectStandardError (Join-Path $results "demo_backend.log")
$started += $backend.Id

if ($reuseDashboard) {
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
if (-not $health) { Fail "The backend did not start; see results\demo_backend.log" }
if (-not (Wait-Http "http://localhost:$FrontendPort/" 90)) { Fail "The dashboard did not start; see results\demo_frontend.log" }

$detector = if ($health.detector) { "$($health.detector.model) on $($health.detector.device)" }
            elseif ($health.detector_error) { "off ($($health.detector_error))" }
            else { "off (it is loaded only with a crowd model)" }
$storage = $health.storage
if ($InMemory) { $storage += " (-InMemory: nothing is saved)" }
elseif (-not $mongoWanted) { $storage += " (no MONGODB_URI in .env: nothing is saved)" }
elseif ($health.storage -ne "mongodb") {
    $storage += " - MongoDB UNREACHABLE (no internet, or this network is not on the Atlas access list): readings are not saved"
    Write-Warning "MongoDB could not be reached; the demo runs with in-memory storage."
}
Write-Host ""
Write-Host "Crowd model : $(if ($health.model_loaded) { $health.model.label } else { $health.model_error })"
Write-Host "Detector    : $detector"
Write-Host "Frames      : $($health.frame_source), zones $($health.zones -join ', ')"
Write-Host "Storage     : $storage"
Write-Host "Telegram    : $telegramNote"
Write-Host ""
Write-Host "Login       : http://localhost:$FrontendPort   (Command Centre after login; the old dashboard is at #/classic)" -ForegroundColor Green
Write-Host "Accounts    : $(if ($accounts.Count) { ($accounts | ForEach-Object { "$($_.username) ($($_.role))" }) -join ', ' } else { 'none yet (see the warning above)' })"
Write-Host "API docs    : http://localhost:$BackendPort/docs"
Write-Host "Logs        : results\demo_backend.log, results\demo_frontend.log   Stop: scripts\stop_demo.ps1"
