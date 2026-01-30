# Production Build Script
# Run this after finishing terrain component development

Write-Host "🏗️  Building Terrain Component for Production..." -ForegroundColor Cyan
Write-Host ""

cd terrain_component/frontend

Write-Host "Installing dependencies..." -ForegroundColor Yellow
npm install

Write-Host ""
Write-Host "Building production bundle (minified, optimized)..." -ForegroundColor Yellow
npm run build

Write-Host ""
cd ..\..

Write-Host "✅ Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Your app is ready to run:" -ForegroundColor Green
Write-Host "  streamlit run app.py" -ForegroundColor White
Write-Host ""
Write-Host "The Streamlit app will use pre-built assets from:" -ForegroundColor Cyan
Write-Host "  terrain_component/frontend/dist/" -ForegroundColor White
Write-Host ""
