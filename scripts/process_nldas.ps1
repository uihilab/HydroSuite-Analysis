
# PowerShell Script: Process NLDAS Data Only (Clip to Regions)
# Usage: .\process_nldas.ps1 -Year 2000 -Month 8

param (
    [Parameter(Mandatory=$true)][int]$Year,
    [Parameter(Mandatory=$true)][int]$Month
)

$ErrorActionPreference = "Stop"
$ScriptsDir = "scripts"

# Configuration
$BaseDir = if ([string]::IsNullOrWhiteSpace($env:HYDRO_DATA_DIR)) { $PSScriptRoot } else { $env:HYDRO_DATA_DIR }
$RawDir = "$BaseDir\nldas"
$ProcLaDir = "data\processed\nldas_louisiana"
$ProcSouthDir = "data\processed\nldas_southern"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " PROCESSING NLDAS: $Year-$Month " -ForegroundColor Cyan
Write-Host " Source: $RawDir" -ForegroundColor Gray
Write-Host "==========================================" -ForegroundColor Cyan

# Run Python processor
python "$ScriptsDir\processing\nldas\clip_nldas_batch.py" $Year $Month

if ($LASTEXITCODE -ne 0) {
    Write-Error "Processing failed!"
}

# Verification
Write-Host "`n[VERIFICATION]" -ForegroundColor Yellow

$monthStr = "{0:D2}" -f $Month
$rawPattern = "NLDAS_FOR0010_D.A$Year$monthStr*.nc"
$procLaPattern = "$Year$monthStr*_louisiana.nc"
$procSouthPattern = "$Year$monthStr*_southern.nc"

$rawCount = (Get-ChildItem -Path $RawDir -Filter $rawPattern).Count
$procLaCount = (Get-ChildItem -Path $ProcLaDir -Filter $procLaPattern).Count
$procSouthCount = (Get-ChildItem -Path $ProcSouthDir -Filter $procSouthPattern).Count

Write-Host "Raw Files (D:):  $rawCount"
Write-Host "Processed LA:    $procLaCount"
Write-Host "Processed South: $procSouthCount"

if ($procLaCount -lt $rawCount -or $procSouthCount -lt $rawCount) {
    Write-Warning "Mismatch detected! Some files were not successfully processed."
} else {
    Write-Host "[OK] Processing Validated" -ForegroundColor Green
}
