$root = Resolve-Path "$PSScriptRoot\.."
$python = Join-Path $root ".venv\Scripts\python.exe"
& $python -m pip install -e (Join-Path $PSScriptRoot "ingestion")
& $python "$PSScriptRoot\app.py"
