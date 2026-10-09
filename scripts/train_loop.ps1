<#
Runs CSRNet training until it is finished, resuming after any crash. Started (hidden) by
scripts\train_full.ps1; you do not run this yourself.

Each pass runs ml.train_csrnet with --resume whenever backend\ml\weights\parta_last.pth exists,
appending its output to results\train_stdout.log. Exit code 0 means training finished (all
epochs, or early stop); anything else is a crash: wait 60 s and resume, at most 20 times.
results\train_status.txt always holds the current state (running / done / failed).
#>
param(
    [int]$Epochs = 400,
    [int]$EarlyStop = 40,
    [int]$BatchSize = 8,
    [int]$Crop = 384,
    [int]$Workers = 1,
    [string]$Data = "..\data\processed\shanghaitech_a"
)
$root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $root "backend")
$status = Join-Path $root "results\train_status.txt"
$stdout = "..\results\train_stdout.log"
$attempts = 0

while ($true) {
    $resume = if (Test-Path "ml\weights\parta_last.pth") { "--resume" } else { "" }
    "running: pass $($attempts + 1) started $(Get-Date -Format 'yyyy-MM-dd HH:mm') $resume" | Set-Content $status
    # cmd /c so the output is appended as plain text (paths here are relative: no spaces).
    cmd /c "..\venv\Scripts\python.exe -u -m ml.train_csrnet --data $Data --name parta --epochs $Epochs --early-stop $EarlyStop --batch-size $BatchSize --crop $Crop --workers $Workers --no-pin-memory --log ..\results\train_log.csv $resume >> $stdout 2>&1"
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        $last = (Get-Content $stdout -Tail 2) -join " | "
        "done: finished $(Get-Date -Format 'yyyy-MM-dd HH:mm') | $last" | Set-Content $status
        break
    }
    $attempts++
    "`n[train_loop] training exited with code $code at $(Get-Date -Format 'yyyy-MM-dd HH:mm'); resuming in 60 s (attempt $attempts of 20)`n" | Add-Content $stdout
    if ($attempts -ge 20) {
        "failed: training crashed $attempts times; see results\train_stdout.log" | Set-Content $status
        break
    }
    Start-Sleep -Seconds 60
}
