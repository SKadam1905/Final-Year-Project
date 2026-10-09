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
if (Test-Path $pidFile) {
    foreach ($id in Get-Content $pidFile) {
        & taskkill /PID $id /T /F 2>$null | Out-Null  # /T: also the python / node child processes
        if ($LASTEXITCODE -eq 0) { Write-Host "stopped $id"; $stopped++ }
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
        & taskkill /PID $proc.ProcessId /T /F 2>$null | Out-Null
        Write-Host "stopped $($proc.Name) (pid $($proc.ProcessId)) on port $port"
        $stopped++
    } else {
        Write-Warning "Port $port is used by $($proc.Name) (pid $($proc.ProcessId)), which is not part of this demo: left running. Stop it with: taskkill /PID $($proc.ProcessId) /T /F"
    }
}
if ($stopped -eq 0) { Write-Host "Nothing to stop: no demo backend or dashboard is running." }
