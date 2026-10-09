<#
Stop the backend and dashboard started by scripts\start_demo.ps1 (a dashboard it reused is left running).

    powershell -ExecutionPolicy Bypass -File scripts\stop_demo.ps1
#>
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "results\demo.pids"
if (-not (Test-Path $pidFile)) {
    Write-Host "Nothing to stop: no demo started by start_demo.ps1 is recorded."
    exit 0
}
foreach ($id in Get-Content $pidFile) {
    & taskkill /PID $id /T /F 2>$null | Out-Null  # /T: also the python / node child processes
    Write-Host "stopped $id"
}
Remove-Item $pidFile
