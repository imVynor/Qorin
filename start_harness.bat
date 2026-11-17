@echo off
echo ========================================================
echo  Starting QORIN Experimentation Harness
echo ========================================================
echo.
echo 1. Launching FastAPI Backend on http://127.0.0.1:8000 ...
start "QORIN Backend API" cmd /k ".\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"

echo 2. Launching Vite Frontend on http://127.0.0.1:5173/harness ...
start "QORIN Frontend UI" cmd /k "npm --prefix web run dev"

echo.
echo Both servers started in separate terminal windows.
echo Open http://127.0.0.1:5173/harness in your browser.

