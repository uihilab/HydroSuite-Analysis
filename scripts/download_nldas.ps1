
# PowerShell Script: Download NLDAS Data Only
# Usage: .\download_nldas.ps1 -Year 2000 -Month 8

param (
    [Parameter(Mandatory=$true)][int]$Year,
    [Parameter(Mandatory=$true)][int]$Month
)

$ErrorActionPreference = "Stop"
$ScriptsDir = "scripts"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " DOWNLOADING NLDAS: $Year-$Month " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Run Node.js downloader (Configured for $HYDRO_DATA_DIR or $PSScriptRoot)
node "$ScriptsDir\retrieval\nldas\get_nldas_monthly.js" $Year $Month

if ($LASTEXITCODE -ne 0) {
    Write-Error "Download failed!"
} else {
    Write-Host "`n[OK] Download Complete" -ForegroundColor Green
}
