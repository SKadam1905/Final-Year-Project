<#
Stop the demo: the backend and dashboard started by scripts\start_demo.ps1, and also this project's
backend (uvicorn app.main:app) or dashboard (Vite) started by hand on ports 8000 / 5173. Another
program on those ports is left running, with the command to stop it.

    powershell -ExecutionPolicy Bypass -File scripts\stop_demo.ps1
#>
param([int]$BackendPort = 8000, [int]$FrontendPort = 5173)
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "results\demo.pids"
$stopped = 0

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
if (Test-Path $pidFile) {
    foreach ($id in Get-Content $pidFile) {
        if ((Test-Ours $id) -and (Stop-Tree $id)) { Write-Host "stopped $id"; $stopped++ }  # /T: also the python / node children
    }
    Remove-Item $pidFile
    Start-Sleep -Seconds 2
}

foreach ($port in $BackendPort, $FrontendPort) {
    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $listener) { continue }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    $ours = $proc.CommandLine -match 'uvicorn.*app\.main:app' -or $proc.CommandLine -like "*nashik-kumbh-stampede-predictor*vite*"
    if ($ours) {
        Stop-Tree $proc.ProcessId | Out-Null
        Write-Host "stopped $($proc.Name) (pid $($proc.ProcessId)) on port $port"
        $stopped++
    } else {
        Write-Warning "Port $port is used by $($proc.Name) (pid $($proc.ProcessId)), which is not part of this demo: left running. Stop it with: taskkill /PID $($proc.ProcessId) /T /F"
    }
}
if ($stopped -eq 0) { Write-Host "Nothing to stop: no demo backend or dashboard is running." }
