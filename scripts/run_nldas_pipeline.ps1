
# PowerShell Script to automate NLDAS Workflow
# Usage: .\run_nldas_pipeline.ps1 -Year 2000 -Month 8

param (
    [Parameter(Mandatory=$true)][int]$Year,
    [Parameter(Mandatory=$true)][int]$Month
)

$ErrorActionPreference = "Stop"

# Paths: raw files under $env:HYDRO_DATA_DIR (script folder if unset)
$BaseDir = if ([string]::IsNullOrWhiteSpace($env:HYDRO_DATA_DIR)) { $PSScriptRoot } else { $env:HYDRO_DATA_DIR }
$RawDir = "$BaseDir\nldas"
$ProcLaDir = "data\processed\nldas_louisiana"
$ProcSouthDir = "data\processed\nldas_southern"
$ScriptsDir = "scripts"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " STARTING NLDAS PIPELINE FOR $Year-$Month " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. DOWNLOAD (Validates >10MB)
Write-Host "`n[STEP 1] Downloading Raw Data..." -ForegroundColor Yellow
node "$ScriptsDir\retrieval\nldas\get_nldas_monthly.js" $Year $Month

if ($LASTEXITCODE -ne 0) {
    Write-Error "Download failed!"
}

# 2. PROCESS (Clips for LA & South + Validates Output)
Write-Host "`n[STEP 2] Processing (Clipping)..." -ForegroundColor Yellow
python "$ScriptsDir\processing\nldas\clip_nldas_batch.py" $Year $Month

if ($LASTEXITCODE -ne 0) {
    Write-Error "Processing failed!"
}

# 3. VERIFY (No Deletion)
Write-Host "`n[STEP 3] Verifying..." -ForegroundColor Yellow

# Count files
$monthStr = "{0:D2}" -f $Month
$rawPattern = "NLDAS_FOR0010_D.A$Year$monthStr*.nc"
$procLaPattern = "$Year$monthStr*_louisiana.nc"
$procSouthPattern = "$Year$monthStr*_southern.nc"

$rawCount = (Get-ChildItem -Path $RawDir -Filter $rawPattern).Count
$procLaCount = (Get-ChildItem -Path $ProcLaDir -Filter $procLaPattern).Count
$procSouthCount = (Get-ChildItem -Path $ProcSouthDir -Filter $procSouthPattern).Count

Write-Host "Raw Files:       $rawCount"
Write-Host "Processed LA:    $procLaCount"
Write-Host "Processed South: $procSouthCount"

if ($procLaCount -lt $rawCount -or $procSouthCount -lt $rawCount) {
    Write-Warning "Mismatch detected! Some files were not successfully processed."
    Write-Warning "Check 'check_processed_sizes.py' to investigate."
} else {
    Write-Host "✓ Processing Complete & Validated" -ForegroundColor Green
}

Write-Host "`nPIPELINE COMPLETE!" -ForegroundColor Cyan
