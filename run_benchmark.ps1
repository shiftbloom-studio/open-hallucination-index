# =============================================================================
# OHI Benchmark Runner Script (PowerShell)
# =============================================================================
# Runs the benchmark inside the Docker container and generates performance charts.
# Charts are automatically saved to ./benchmark_results/charts/
#
# Usage:
#   .\run_benchmark.ps1                       # Run with defaults
#   .\run_benchmark.ps1 -Strategy mcp_enhanced
#   .\run_benchmark.ps1 -Limit 50             # Run only 50 test cases
# =============================================================================

param(
    [string]$Strategy = "",
    [int]$Limit = 0,
    [switch]$Help
)

$ContainerName = "ohi-benchmark"
$OutputDir = ".\benchmark_results"

# Colors
$Green = "Green"
$Blue = "Cyan"
$Yellow = "Yellow"
$Red = "Red"

function Write-Header {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor $Blue
    Write-Host "║              OHI Benchmark Runner                            ║" -ForegroundColor $Blue
    Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor $Blue
}

function Write-Section($text) {
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor $Blue
}

if ($Help) {
    Write-Host @"
OHI Benchmark Runner

Usage:
    .\run_benchmark.ps1                       # Run with defaults
    .\run_benchmark.ps1 -Strategy mcp_enhanced
    .\run_benchmark.ps1 -Limit 50             # Run only 50 test cases
    .\run_benchmark.ps1 -Help                 # Show this help

Strategies:
    - vector_semantic
    - graph_exact
    - hybrid
    - cascading
    - mcp_enhanced
    - adaptive
"@
    exit 0
}

Write-Header

# Check if container is running
$running = docker ps --format '{{.Names}}' | Select-String -Pattern "^$ContainerName$"
if (-not $running) {
    Write-Host "⚠ Benchmark container not running. Starting services..." -ForegroundColor $Yellow
    docker compose up -d benchmark-runner
    Start-Sleep -Seconds 3
}

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path "$OutputDir\charts" | Out-Null

# Build command arguments
$args = @()
if ($Strategy) {
    $args += "--strategy"
    $args += $Strategy
}
if ($Limit -gt 0) {
    $args += "--limit"
    $args += $Limit.ToString()
}

# Run benchmark
Write-Host ""
Write-Host "🚀 Starting benchmark..." -ForegroundColor $Green
Write-Section

$argsString = $args -join " "
if ($argsString) {
    docker exec -it $ContainerName python -m benchmark $args
} else {
    docker exec -it $ContainerName python -m benchmark
}

# Results
Write-Host ""
Write-Host "📊 Benchmark complete!" -ForegroundColor $Green
Write-Section

# List generated charts
$charts = Get-ChildItem -Path "$OutputDir\charts\*.png" -ErrorAction SilentlyContinue
if ($charts) {
    Write-Host ""
    Write-Host "📈 Generated Performance Charts:" -ForegroundColor $Green
    foreach ($chart in $charts | Sort-Object LastWriteTime -Descending | Select-Object -First 6) {
        Write-Host "  → $($chart.Name)" -ForegroundColor White
    }
    Write-Host ""
    Write-Host "Charts saved to: $OutputDir\charts\" -ForegroundColor $Blue
}

# List reports
Write-Host ""
Write-Host "📄 Generated Reports:" -ForegroundColor $Green
$reports = Get-ChildItem -Path "$OutputDir\ohi_benchmark_*" -Include "*.json","*.csv","*.md","*.html" -ErrorAction SilentlyContinue
foreach ($report in $reports | Sort-Object LastWriteTime -Descending | Select-Object -First 8) {
    Write-Host "  → $($report.Name)" -ForegroundColor White
}

Write-Host ""
Write-Host "✅ Done!" -ForegroundColor $Green
