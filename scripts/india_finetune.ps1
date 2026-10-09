<#
India fine-tuning in one go (Phase 5, step 6; see results\INDIA_FINETUNE.md first).

    powershell -ExecutionPolicy Bypass -File scripts\india_finetune.ps1 -TestVideos shirdi_01 -ValVideos pandharpur_02

1. Builds density maps from the CVAT head points and splits BY VIDEO: the -TestVideos and
   -ValVideos never share a video with training (ml.prepare_dataset refuses overlaps).
2. Evaluates the starting weights on the India test videos      -> results\india_before\
3. Fine-tunes from them with a low learning rate (1e-6, 60 epochs; best epoch chosen on the
   validation videos only)                                      -> backend\ml\weights\india_ft_best.pth
4. Evaluates the fine-tuned weights on the same test videos     -> results\india_after\
5. Compares before/after, overall and per video                 -> results\india_finetune.csv/.md
   and stores the test MAE/RMSE in the fine-tuned checkpoint's provenance.
Each model is evaluated on the test videos exactly once.
#>
param(
    [Parameter(Mandatory = $true)][string]$TestVideos,  # comma-separated video ids, e.g. shirdi_01
    [Parameter(Mandatory = $true)][string]$ValVideos,
    [string]$Init = "",  # starting weights (default: CSRNET_WEIGHTS from .env)
    [int]$Epochs = 60,
    [double]$Lr = 1e-6
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "venv\Scripts\python.exe"
Set-Location (Join-Path $root "backend")

if (-not $Init) {
    $line = Select-String -Path (Join-Path $root ".env") -Pattern '^CSRNET_WEIGHTS=(.+)$' | Select-Object -First 1
    if (-not $line) { throw "No -Init given and no CSRNET_WEIGHTS in .env" }
    $Init = Join-Path $root $line.Matches[0].Groups[1].Value.Trim()
}
if (-not (Test-Path $Init)) { throw "Starting weights not found: $Init" }
Write-Host "Starting weights: $Init"

& $python -m ml.prepare_dataset india --test-videos $TestVideos --val-videos $ValVideos
if ($LASTEXITCODE) { throw "prepare_dataset failed" }
$data = "..\data\processed\india"

& $python -m ml.evaluate --weights $Init --data $data --out ..\results\india_before
if ($LASTEXITCODE) { throw "evaluation before fine-tuning failed" }

& $python -u -m ml.train_csrnet --data $data --init $Init --name india_ft --epochs $Epochs --lr $Lr `
    --log ..\results\india_train_log.csv
if ($LASTEXITCODE) { throw "fine-tuning failed" }

& $python -m ml.evaluate --weights ml\weights\india_ft_best.pth --data $data --out ..\results\india_after
if ($LASTEXITCODE) { throw "evaluation after fine-tuning failed" }

& $python ..\scripts\india_compare.py ..\results\india_before\results.csv ..\results\india_after\results.csv `
    --index "$data\index.csv" --weights ml\weights\india_ft_best.pth
Write-Host "Done: results\india_finetune.md"
