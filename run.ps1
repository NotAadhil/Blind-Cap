# Blind Cap - 1-Click PowerShell Launcher
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       BLIND CAP - ACCESSIBLE VISION & 3D AUDIO SYSTEM" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# Create Virtual Environment if not present
if (-not (Test-Path ".venv")) {
    Write-Host "[1/3] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate venv
$activateScript = ".venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
}

# Install Requirements
Write-Host "[2/3] Checking dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet --disable-pip-version-check

# Check OpenVINO model
if (-not (Test-Path "models\yolo26m_480_openvino\yolo26m.xml")) {
    Write-Host "[3/3] Exporting YOLO26m model for OpenVINO GPU/CPU acceleration..." -ForegroundColor Yellow
    python export_model.py
} else {
    Write-Host "[3/3] Model ready." -ForegroundColor Green
}

Write-Host "`nStarting Blind Cap in 60 FPS Viewfinder Mode...`n" -ForegroundColor Green
python main.py
