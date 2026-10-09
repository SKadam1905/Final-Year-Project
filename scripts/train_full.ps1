<#
Full CSRNet training on ShanghaiTech Part A, in the background (Phase 5A, step 6).

    powershell -ExecutionPolicy Bypass -File scripts\train_full.ps1     # start, or resume after a reboot
    powershell -ExecutionPolicy Bypass -File scripts\check_training.ps1 # progress, best val MAE, time left
    Stop-Process -Id (Get-Content results\train.pid)                    # stop the loop (then stop python too:
                                                                        #  check_training.ps1 shows its PID)

Starts scripts\train_loop.ps1 in a hidden PowerShell that does not belong to this terminal, so
closing the terminal does not stop training. The loop resumes automatically after a crash. After
a reboot or a stop, run this script again: it continues from backend\ml\weights\parta_last.pth.

Training: up to 400 epochs, stopping early once the validation MAE has not improved for 40
epochs. Batch 8 x 384 px crops with mixed precision (results\training_vram.md lists the largest
that fit: 8 x 512 or 12 x 384). One row per epoch goes to results\train_log.csv (epoch,
train_loss, val_loss, val_mae, val_rmse, lr, seconds). The best checkpoint by validation MAE is
backend\ml\weights\parta_best.pth; the last is parta_last.pth (both git-ignored). The test set is
never used here.

Keep the laptop plugged in, and set "When plugged in, put my device to sleep after" to Never
(Settings > System > Power), or Windows pauses training while it sleeps.
#>
param(
    [int]$Epochs = 400,
    [int]$EarlyStop = 40,
    [int]$BatchSize = 8,
    [int]$Crop = 384,
    [int]$Workers = 1,  # one loader worker, no pinned memory: this laptop has 7.6 GB RAM
    [string]$Data = "..\data\processed\shanghaitech_a"  # relative to backend\
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$results = Join-Path $root "results"
New-Item -ItemType Directory -Force $results | Out-Null

if (-not (Test-Path (Join-Path $root "backend\$Data\index.csv"))) {
    throw "No processed dataset at $Data (from backend\). Run first: cd backend; ..\venv\Scripts\python.exe -m ml.prepare_dataset shanghaitech"
}
$pidFile = Join-Path $results "train.pid"
if ((Test-Path $pidFile) -and (Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue)) {
    throw "Training is already running (loop PID $(Get-Content $pidFile)). Check it with: powershell -ExecutionPolicy Bypass -File scripts\check_training.ps1"
}
if (Test-Path (Join-Path $root "backend\ml\weights\parta_last.pth")) { Write-Host "Resuming from backend\ml\weights\parta_last.pth" }

$loop = Join-Path $PSScriptRoot "train_loop.ps1"
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$loop`" -Epochs $Epochs -EarlyStop $EarlyStop " +
             "-BatchSize $BatchSize -Crop $Crop -Workers $Workers -Data `"$Data`""
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru
$process.Id | Set-Content $pidFile
Write-Host "Training started in the background (loop PID $($process.Id))."
Write-Host "Check it with: powershell -ExecutionPolicy Bypass -File scripts\check_training.ps1"
