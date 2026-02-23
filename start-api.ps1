param(
    [int]$Port = 8000,
    [string]$BindAddress = "0.0.0.0",
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Nao encontrei $pythonExe. Crie a venv primeiro: py -3.12 -m venv .venv"
}

# Limpa proxies/flags que podem atrapalhar execucao local.
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:GIT_HTTP_PROXY = ""
$env:GIT_HTTPS_PROXY = ""
$env:PIP_NO_INDEX = ""

$reloadArg = ""
if ($Reload) {
    $reloadArg = "--reload"
}

Write-Host "Iniciando API em http://localhost:$Port ..."
& $pythonExe -m uvicorn api.main:app --host $BindAddress --port $Port $reloadArg
