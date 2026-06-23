# Run EIDOS live eval from the eidos repo (works even if your shell cwd is elsewhere).
$ErrorActionPreference = "Stop"
$EidosRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $EidosRoot

if (-not $env:GROQ_API_KEY -and -not $env:OPENAI_API_KEY) {
    Write-Host "Note: GROQ_API_KEY is not set. Live eval will SKIP (not an import error)." -ForegroundColor Yellow
    Write-Host "  set GROQ_API_KEY=gsk_your_key_here" -ForegroundColor Yellow
}

py run_live_eval.py @args
