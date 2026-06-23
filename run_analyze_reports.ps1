# Analyze EIDOS live eval reports from the eidos repo (works if shell cwd is elsewhere).
$ErrorActionPreference = "Stop"
$EidosRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $EidosRoot

py run_analyze_reports.py @args
