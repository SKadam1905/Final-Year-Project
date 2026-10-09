<#
Training progress at a glance (Phase 5A, step 6).

    powershell -ExecutionPolicy Bypass -File scripts\check_training.ps1

Shows whether the training loop and python are running, the status line, the latest epochs, the
best validation MAE so far, and the time left: to the 400-epoch limit, and to the early stop
(40 epochs without improvement), whichever comes first. "done:" in the status means finished.
#>
param([int]$Last = 8)
$root = Split-Path -Parent $PSScriptRoot
$results = Join-Path $root "results"
$log = Join-Path $results "train_log.csv"
$pidFile = Join-Path $results "train.pid"
$statusFile = Join-Path $results "train_status.txt"

$loopAlive = (Test-Path $pidFile) -and [bool](Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue)
$trainer = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'ml\.train_csrnet' -and $_.CommandLine -match 'parta' } | Select-Object -First 1
Write-Host ("Loop running:   {0}" -f $(if ($loopAlive) { "yes (PID $(Get-Content $pidFile))" } else { "no" }))
Write-Host ("Python running: {0}" -f $(if ($trainer) { "yes (PID $($trainer.ProcessId))" } else { "no" }))
if (Test-Path $statusFile) { Write-Host "Status:         $(Get-Content $statusFile -Raw)".TrimEnd() }

if (-not (Test-Path $log)) { Write-Host "No results\train_log.csv yet (the first epoch takes about a minute)."; exit }
$rows = Import-Csv $log | Where-Object { $_.val_mae -ne "nan" }
if (-not $rows) { Write-Host "No finished epoch yet."; exit }
$rows | Select-Object -Last $Last | Format-Table epoch, train_loss, val_loss, val_mae, val_rmse, lr, seconds -AutoSize | Out-String | Write-Host

$best = $rows | Sort-Object { [double]$_.val_mae } | Select-Object -First 1
$epoch = [int]($rows | Select-Object -Last 1).epoch
$recent = $rows | Select-Object -Last 10 | ForEach-Object { [double]$_.seconds }
$perEpoch = ($recent | Measure-Object -Average).Average
$maxEpochs = 400; $patience = 40
$cmd = if ($trainer) { $trainer.CommandLine } else { "" }
if ($cmd -match '--epochs (\d+)') { $maxEpochs = [int]$Matches[1] }
if ($cmd -match '--early-stop (\d+)') { $patience = [int]$Matches[1] }
$leftLimit = [math]::Max(0, $maxEpochs - $epoch)
$leftEarly = [math]::Min($leftLimit, [math]::Max(0, [int]$best.epoch + $patience - $epoch))
Write-Host ("Best val MAE so far: {0:N2} at epoch {1} (RMSE {2:N2})" -f [double]$best.val_mae, $best.epoch, [double]$best.val_rmse)
Write-Host ("Epochs done: {0} of at most {1}; {2:N0} s per epoch (last 10)" -f $epoch, $maxEpochs, $perEpoch)
Write-Host ("Earliest finish: {0} ({1} epochs, if the val MAE never improves again: early stop after {2} epochs without improvement)" -f (Get-Date).AddSeconds($leftEarly * $perEpoch).ToString("ddd HH:mm"), $leftEarly, $patience)
Write-Host ("Latest finish:   {0} ({1} epochs, the {2}-epoch limit)" -f (Get-Date).AddSeconds($leftLimit * $perEpoch).ToString("ddd HH:mm"), $leftLimit, $maxEpochs)
Write-Host "Each improvement moves the earliest finish later. Done when Status starts with 'done:'."
