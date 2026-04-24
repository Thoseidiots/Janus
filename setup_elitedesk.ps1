# Run this in PowerShell on the EliteDesk via Remote Desktop
# It checks hardware, installs Python deps, and starts TTS training

Write-Host "=== EliteDesk Hardware ===" -ForegroundColor Cyan
$cpu = Get-WmiObject Win32_Processor
Write-Host "CPU: $($cpu.Name)"
Write-Host "Cores: $($cpu.NumberOfCores) physical / $($cpu.NumberOfLogicalProcessors) logical"
$ram = [math]::Round((Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory/1GB, 1)
Write-Host "RAM: $ram GB"
$gpu = Get-WmiObject Win32_VideoController | Select-Object -First 2
$gpu | ForEach-Object { Write-Host "GPU: $($_.Name)" }

Write-Host ""
Write-Host "=== Python Check ===" -ForegroundColor Cyan
python --version 2>&1
pip --version 2>&1

Write-Host ""
Write-Host "=== Installing dependencies ===" -ForegroundColor Cyan
pip install torch numpy scipy soundfile --quiet

Write-Host ""
Write-Host "=== CUDA Check ===" -ForegroundColor Cyan
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host ""
Write-Host "=== Cloning/Syncing Janus repo ===" -ForegroundColor Cyan
# Option A: if git is installed
if (Get-Command git -ErrorAction SilentlyContinue) {
    if (Test-Path "C:\Janus") {
        Set-Location "C:\Janus"
        git pull origin main
    } else {
        git clone https://github.com/Thoseidiots/Janus.git C:\Janus
        Set-Location "C:\Janus"
    }
    Write-Host "Repo ready at C:\Janus"
} else {
    Write-Host "Git not found. Install from https://git-scm.com or copy files manually."
}

Write-Host ""
Write-Host "=== Ready to train ===" -ForegroundColor Green
Write-Host "Run: python C:\Janus\janus_tts_v2.py"
