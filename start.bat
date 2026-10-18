@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto :python_failed
)
if not exist ".venv\.realityforge-installed" (
  .venv\Scripts\python.exe -m pip install -r requirements-lock.txt
  if errorlevel 1 goto :dependencies_failed
  type nul > .venv\.realityforge-installed
)
echo.
.venv\Scripts\python.exe -m forge serve %*
if errorlevel 1 goto :server_failed
exit /b 0
:python_failed
echo.
echo Could not create the Python environment. Install Python 3.11 or newer with Add Python to PATH enabled.
pause
exit /b 1
:dependencies_failed
echo.
echo Dependency installation failed. Check your internet connection and the error above, then try again.
pause
exit /b 1
:server_failed
echo.
echo The server could not start. See the specific error above.
echo To use another port, run: npm run dev -- --port 8001
pause
exit /b 1
