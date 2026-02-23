param(
    [int]$Port = 8502,
    [string]$BindAddress = "0.0.0.0"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dashboardApp = Join-Path $projectRoot "dashboard\dashapp.py"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Nao encontrei $pythonExe. Crie a venv primeiro: py -3.12 -m venv .venv"
}

if (-not (Test-Path $dashboardApp)) {
    Write-Error "Nao encontrei $dashboardApp."
}

# Limpa proxies/flags que atrapalham a execucao local.
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:GIT_HTTP_PROXY = ""
$env:GIT_HTTPS_PROXY = ""
$env:PIP_NO_INDEX = ""

Write-Host "Iniciando dashboard em http://localhost:$Port ..."
& $pythonExe $dashboardApp --host $BindAddress --port $Port
