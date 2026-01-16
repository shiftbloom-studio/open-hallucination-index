@echo off
setlocal
set "ROOT=%~dp0.."
set "PY_DIR=%ROOT%\.python"
set "VENV_DIR=%ROOT%\.venv"
set "PY_VER=3.11.7"
set "PY_ZIP=python-%PY_VER%-embed-amd64.zip"
set "PY_URL=https://www.python.org/ftp/python/%PY_VER%/%PY_ZIP%"
set "PY_EXE=%PY_DIR%\python.exe"

if not exist "%PY_EXE%" (
	echo [setup] Bootstrapping local Python %PY_VER%...
	if not exist "%PY_DIR%" mkdir "%PY_DIR%"
	powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_DIR%\%PY_ZIP%'"
	powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%PY_DIR%\%PY_ZIP%' -DestinationPath '%PY_DIR%' -Force"
	del "%PY_DIR%\%PY_ZIP%" >nul 2>&1
	powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Path '%PY_DIR%' -Filter 'python*._pth' | ForEach-Object { (Get-Content $_.FullName) -replace '^#\s*import\s+site', 'import site' | Set-Content $_.FullName }"
	powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PY_DIR%\get-pip.py'"
	"%PY_EXE%" "%PY_DIR%\get-pip.py"
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
	echo [setup] Creating venv...
	"%PY_EXE%" -m pip install --upgrade pip virtualenv
	"%PY_EXE%" -m virtualenv "%VENV_DIR%"
)

"%VENV_DIR%\Scripts\python.exe" -m pip install -e "%~dp0ingestion"
"%VENV_DIR%\Scripts\python.exe" "%~dp0app.py"
endlocal
