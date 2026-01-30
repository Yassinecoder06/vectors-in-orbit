# Dev Mode Setup Script for Terrain Component
# Run this to enable development mode with hot reload

Write-Host "🔧 Starting Terrain Component Development Mode..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Setting environment variable: TERRAIN_CANVAS_DEV_URL=http://0.0.0.0:5175" -ForegroundColor Yellow

# Set the environment variable for this session
$env:TERRAIN_CANVAS_DEV_URL = "http://0.0.0.0:5175"

Write-Host ""
Write-Host "Split your terminal into 2 tabs/panes:" -ForegroundColor Green
Write-Host ""
Write-Host "TAB 1 (Vite Dev Server - RUN THIS FIRST):" -ForegroundColor Yellow
Write-Host "  cd terrain_component/frontend" -ForegroundColor White
Write-Host "  npm run dev -- --host 0.0.0.0 --port 5175" -ForegroundColor White
Write-Host ""
Write-Host "TAB 2 (Streamlit App - RUN THIS SECOND):" -ForegroundColor Yellow
Write-Host "  streamlit run app.py" -ForegroundColor White
Write-Host ""
Write-Host "The terrain component will hot reload on file changes in src/" -ForegroundColor Green
Write-Host ""
Write-Host "To exit dev mode, close the terminals and build production:" -ForegroundColor Yellow
Write-Host "  cd terrain_component/frontend" -ForegroundColor White
Write-Host "  npm run build" -ForegroundColor White
Write-Host ""
