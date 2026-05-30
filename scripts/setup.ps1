# Instalacja zaleznosci (SSL / Windows)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:PIP_CONFIG_FILE = Join-Path (Get-Location) "pip.ini"
$th = "--trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org"

try {
    $cert = & python -c "import certifi; print(certifi.where())"
    if ($cert -and (Test-Path $cert)) {
        $env:SSL_CERT_FILE = $cert
        $env:REQUESTS_CA_BUNDLE = $cert
    }
} catch { }

if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "Tworze venv (--system-site-packages)..."
    & python -m venv venv --system-site-packages
}

$py = Join-Path (Get-Location) "venv\Scripts\python.exe"
Write-Host "Instalacja requirements.txt..."
cmd /c "`"$py`" -m pip install $th -r requirements.txt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nGotowe."
Write-Host "  .\venv\Scripts\activate"
Write-Host "  python scripts\train.py"
