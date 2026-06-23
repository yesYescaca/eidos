$ErrorActionPreference = "Stop"
$EidosRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $EidosRoot
py run_compare_ablation.py @args
