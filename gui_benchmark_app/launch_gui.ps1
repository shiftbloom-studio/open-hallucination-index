$root = Resolve-Path "$PSScriptRoot\.."
$python = Join-Path $root ".venv\Scripts\python.exe"
$appDir = Resolve-Path $PSScriptRoot
& $python -m pip install -e (Join-Path $appDir "benchmark")
& $python "$PSScriptRoot\app.py"
