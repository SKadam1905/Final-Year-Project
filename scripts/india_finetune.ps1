<#
India fine-tuning, overnight, after the CVAT exports are imported (results\ANNOTATION_GUIDE.md).

    powershell -ExecutionPolicy Bypass -File scripts\india_finetune.ps1

or, to start it and close the terminal (it keeps running; check the log in the morning):

    Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File scripts\india_finetune.ps1" -WindowStyle Minimized

1. Rebuilds the India dataset from the imported head points (tools.india_dataset build), keeping the
   split in data\india_dataset\split.csv: whole videos, and the test locations never trained on.
2. Evaluates the starting weights on the India test crops            -> results\india_before\
3. Fine-tunes from them with a low learning rate (1e-6), up to 100 epochs, stopping after 20
   without a better validation MAE; the best epoch is chosen on the
   validation videos only                                            -> backend\ml\weights\india_ft_best.pth
4. Evaluates the fine-tuned weights on the same test crops           -> results\india_after\
5. Compares before/after overall, per test location and per video    -> results\india_finetune.csv/.md
   and stores the test MAE/RMSE in the fine-tuned checkpoint's provenance.
Each model is evaluated on the test crops exactly once. Every step's output goes to
results\india_finetune_run.log (watch it with: Get-Content results\india_finetune_run.log -Wait -Tail 20).
Close Chrome and other heavy programs first: training needs most of the 7.6 GB of RAM.
#>
param(
    [string]$TestVideos = "",  # override the test videos (comma-separated ids); normally leave empty
    [string]$ValVideos = "",
    [string]$Init = "",  # starting weights (default: CSRNET_WEIGHTS from .env)
    [int]$Epochs = 100,
    [int]$EarlyStop = 20,
    [double]$Lr = 1e-6
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $root "backend")
# Relative paths from backend\ hold no spaces, so cmd /c can redirect the output without extra quoting.
$python = "..\venv\Scripts\python.exe"
$log = "..\results\india_finetune_run.log"
$data = "..\data\processed\india"

if (-not (Test-Path "..\data\india_dataset\split.csv") -and -not $TestVideos) {
    throw "No data\india_dataset\split.csv: import the CVAT exports first (python -m tools.india_dataset import)"
}
if (-not $Init) {
    $line = Select-String -Path (Join-Path $root ".env") -Pattern '^CSRNET_WEIGHTS=(.+)$' | Select-Object -First 1
    if (-not $line) { throw "No -Init given and no CSRNET_WEIGHTS in .env" }
    $Init = Join-Path $root $line.Matches[0].Groups[1].Value.Trim()
}
if (-not (Test-Path $Init)) { throw "Starting weights not found: $Init" }
$Init = Resolve-Path -Relative $Init

function Step([string]$Title, [string]$Arguments) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "$stamp  $Title"
    Add-Content -Path $log -Value "`r`n=== $stamp  $Title`r`n$python $Arguments" -Encoding utf8
    cmd /c "$python $Arguments >> $log 2>&1"
    if ($LASTEXITCODE) { throw "$Title failed (exit code $LASTEXITCODE); see results\india_finetune_run.log" }
}

Set-Content -Path $log -Value "India fine-tuning run, starting weights $Init" -Encoding utf8
$override = ""
if ($TestVideos) { $override = "--test-videos $TestVideos --val-videos $ValVideos" }
Step "1/5 build the India dataset (split by video and location)" "-m tools.india_dataset build --out $data $override"
Step "2/5 evaluate the starting weights on the test crops" "-m ml.evaluate --weights $Init --data $data --out ..\results\india_before"
Step "3/5 fine-tune (log: results\india_train_log.csv)" ("-u -m ml.train_csrnet --data $data --init $Init --name india_ft " +
    "--epochs $Epochs --early-stop $EarlyStop --lr $Lr --workers 1 --no-pin-memory --log ..\results\india_train_log.csv")
Step "4/5 evaluate the fine-tuned weights on the same test crops" "-m ml.evaluate --weights ml\weights\india_ft_best.pth --data $data --out ..\results\india_after"
Step "5/5 compare before and after, per location" ("..\scripts\india_compare.py ..\results\india_before\results.csv " +
    "..\results\india_after\results.csv --index $data\index.csv --split ..\data\india_dataset\split.csv " +
    "--weights ml\weights\india_ft_best.pth")
Get-Content "..\results\india_finetune.md" | Write-Host
Write-Host "Done: results\india_finetune.md"
