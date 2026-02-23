param(
    [int]$ApiPort = 8000,
    [int]$DashboardPort = 8502,
    [string]$BindAddress = "0.0.0.0",
    [switch]$ApiReload
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$apiScript = Join-Path $projectRoot "start-api.ps1"
$dashboardScript = Join-Path $projectRoot "start-dashboard.ps1"

if (-not (Test-Path $apiScript)) {
    Write-Error "Nao encontrei $apiScript."
}
if (-not (Test-Path $dashboardScript)) {
    Write-Error "Nao encontrei $dashboardScript."
}

$apiArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$apiScript`"",
    "-Port", "$ApiPort",
    "-BindAddress", "$BindAddress"
)
if ($ApiReload) {
    $apiArgs += "-Reload"
}

$dashboardArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$dashboardScript`"",
    "-Port", "$DashboardPort",
    "-BindAddress", "$BindAddress"
)

Write-Host "Subindo API em http://localhost:$ApiPort ..."
Start-Process powershell -ArgumentList $apiArgs -WorkingDirectory $projectRoot | Out-Null

Write-Host "Subindo dashboard em http://localhost:$DashboardPort ..."
Start-Process powershell -ArgumentList $dashboardArgs -WorkingDirectory $projectRoot | Out-Null

Write-Host "Servicos iniciados."
