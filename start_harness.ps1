Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting QORIN Experimentation Harness" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

Write-Host "`n1. Launching FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Yellow
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", ".\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"

Write-Host "2. Launching Vite Frontend on http://127.0.0.1:5173/harness ..." -ForegroundColor Yellow
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "npm --prefix web run dev"

Write-Host "`nBoth services are starting." -ForegroundColor Green
Write-Host "Open http://127.0.0.1:5173/harness in your browser.`n" -ForegroundColor White

