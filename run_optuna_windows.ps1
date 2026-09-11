param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("shakespeare", "femnist")]
    [string]$Dataset,

    [int]$StartIndex = 0,
    [int]$EndIndex = 23,

    [string]$SearchConfig = "configs/controlled_search_grid.yaml"
)

$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
Set-Location $ProjectDir

$LogDir = Join-Path $ProjectDir "logs\$Dataset"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "========================================"
Write-Host "Local Optuna search"
Write-Host "Dataset: $Dataset"
Write-Host "Cases: $StartIndex -> $EndIndex"
Write-Host "Project: $ProjectDir"
Write-Host "========================================"

Write-Host ""
Write-Host "Python environment"
python --version

python -c "import sys, torch, optuna; print('Python:', sys.executable); print('Torch:', torch.__version__); print('Optuna:', optuna.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

if ($LASTEXITCODE -ne 0) {
    throw "Python environment validation failed."
}

Write-Host ""
Write-Host "Starting experiment cases..."

for ($Index = $StartIndex; $Index -le $EndIndex; $Index++) {

    Write-Host ""
    Write-Host "========================================"
    Write-Host "Resolving case index $Index"
    Write-Host "========================================"

    $CaseInfo = python run_case_index.py `
        --dataset $Dataset `
        --index $Index

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to resolve case index $Index"
    }

    $Parts = $CaseInfo.Trim() -split "\s+"

    if ($Parts.Count -lt 3) {
        throw "Unexpected run_case_index.py output: $CaseInfo"
    }

    $ResolvedDataset = $Parts[0]
    $Case = $Parts[1]
    $Algorithm = $Parts[2]

    Write-Host "Dataset   : $ResolvedDataset"
    Write-Host "Case      : $Case"
    Write-Host "Algorithm : $Algorithm"

    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

    $OutFile = Join-Path $LogDir `
        "${Index}_${Algorithm}_${Case}_${Timestamp}.out"

    $ErrFile = Join-Path $LogDir `
        "${Index}_${Algorithm}_${Case}_${Timestamp}.err"

    Write-Host "OUT: $OutFile"
    Write-Host "ERR: $ErrFile"

    $Arguments = @(
        "run_optuna_dirichlet_search.py",
        "--dataset", $ResolvedDataset,
        "--case", $Case,
        "--algorithm", $Algorithm,
        "--search-config", $SearchConfig
    )

    $Process = Start-Process `
        -FilePath "python" `
        -ArgumentList $Arguments `
        -RedirectStandardOutput $OutFile `
        -RedirectStandardError $ErrFile `
        -NoNewWindow `
        -Wait `
        -PassThru

    if ($Process.ExitCode -ne 0) {
        Write-Warning "Case $Index failed: $Algorithm / $Case"
        Write-Warning "See: $ErrFile"

        # Continue with the next case rather than losing the entire desktop run.
        continue
    }

    Write-Host "Case $Index completed successfully."
}

Write-Host ""
Write-Host "========================================"
Write-Host "$Dataset search batch finished"
Write-Host "========================================"