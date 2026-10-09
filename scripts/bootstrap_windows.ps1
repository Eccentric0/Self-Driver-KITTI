param(
    [string]$CondaEnvironment = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if ($CondaEnvironment) {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        throw "未找到 conda。请激活现有 Python 环境后直接运行此脚本。"
    }
    conda run -n $CondaEnvironment python tools\check_environment.py
} else {
    python tools\check_environment.py
}

if ($LASTEXITCODE -ne 0) {
    throw "环境检查失败，退出码：$LASTEXITCODE"
}

Write-Host "环境检查通过；项目脚本可直接从 src 导入本地代码。"
