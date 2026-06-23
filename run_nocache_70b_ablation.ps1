# 70B mixed N=50 ablation: fresh Groq calls (--no-cache) vs cached baseline.
# Requires GROQ_API_KEY. Saves a separate report (does not overwrite cached JSON).
$ErrorActionPreference = "Stop"
$EidosRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $EidosRoot

if (-not $env:GROQ_API_KEY) {
    Write-Host "ERROR: GROQ_API_KEY is not set." -ForegroundColor Red
    Write-Host '  $env:GROQ_API_KEY="gsk_your_key_here"' -ForegroundColor Yellow
    exit 1
}

$MODES = @("llm_alone", "llm_cot", "llm_reflection", "eidos_belief")
$Out = "eval/eidos_eval/reports/live_mixed_llama-3.3-70b-versatile_nocache_report.json"
$Cached = "eval/eidos_eval/reports/live_mixed_llama-3.3-70b-versatile_report.json"

Write-Host "Running 70B mixed eval with --no-cache (4 modes x 50 questions)..." -ForegroundColor Cyan
py run_live_eval.py `
  --provider groq `
  --model llama-3.3-70b-versatile `
  --mixed `
  --no-cache `
  --modes llm_alone llm_cot llm_reflection eidos_belief `
  --out $Out

Write-Host "`nComparing cached vs no-cache..." -ForegroundColor Cyan
py run_compare_ablation.py $Cached $Out `
  --metrics task_accuracy ambiguous_safe_rate misconception_commit_ti_rate `
  --out eval/eidos_eval/reports/nocache_ablation_70b_mixed.json

Write-Host "`nDone. Patching paper Figure 8..." -ForegroundColor Cyan
py update_paper_nocache_figure.py
Write-Host "Done. Review docs/EIDOS_Research_Paper.html Figure 8 + Table 13." -ForegroundColor Green
