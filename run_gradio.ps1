# Script para ejecutar la interfaz Gradio

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Activando entorno virtual..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

& .\venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Instalando/actualizando Gradio..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

pip install -q gradio

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Iniciando aplicacion Gradio..." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Abre en navegador: http://localhost:7860" -ForegroundColor Yellow
Write-Host "Para detener: Ctrl + C en la terminal" -ForegroundColor Yellow
Write-Host ""

python gradio_app.py
