$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (-not $env:QUANT_PLATFORM_REDIS_URL) {
    $env:QUANT_PLATFORM_REDIS_URL = "redis://localhost:6379/0"
}

python -m quant_platform.jobs.workers @args
exit $LASTEXITCODE
