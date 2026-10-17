@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto :failed
)
if not exist ".venv\.realityforge-installed" (
  .venv\Scripts\python.exe -m pip install -r requirements-lock.txt
  if errorlevel 1 goto :failed
  type nul > .venv\.realityforge-installed
)
echo.
echo RealityForge is starting at http://127.0.0.1:8000
echo Keep this window open. Press Ctrl+C to stop the generator.
echo.
.venv\Scripts\python.exe -m forge serve
if errorlevel 1 goto :failed
exit /b 0
:failed
echo.
echo Startup failed. Install Python 3.11 or newer with Add Python to PATH enabled.
echo If port 8000 is occupied, run: .venv\Scripts\python.exe -m forge serve --port 8001
pause
exit /b 1
