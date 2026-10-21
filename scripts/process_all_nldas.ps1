
# PowerShell Script: Process ALL NLDAS Data
# Usage: .\process_all_nldas.ps1

$ErrorActionPreference = "Stop"
$ScriptsDir = "scripts"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " STARTING BULK NLDAS PROCESSING " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Run Python processor with 0 0 (Process ALL)
python "$ScriptsDir\processing\nldas\clip_nldas_batch.py" 0 0

if ($LASTEXITCODE -ne 0) {
    Write-Error "Processing failed!"
} else {
    Write-Host "`n[OK] Bulk Processing Complete" -ForegroundColor Green
}
