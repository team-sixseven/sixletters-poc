# Run from an elevated Windows PowerShell. Changes only MLflow TCP/5000 rules.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [System.Net.IPAddress[]]$ListenAddress,
    [string]$Distribution = "Ubuntu-24.04",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Open Windows PowerShell as Administrator, then run this script again."
}

if (-not $Remove) {
    $addressOutput = & wsl.exe -d $Distribution -- ip -4 -o addr show dev eth0
    if ($LASTEXITCODE -ne 0) { throw "Cannot read the WSL eth0 address." }
    $matched = [regex]::Match(($addressOutput -join " "), 'inet\s+(\d+\.\d+\.\d+\.\d+)/')
    if (-not $matched.Success) { throw "No WSL IPv4 address found." }
    $wslAddress = $matched.Groups[1].Value
    $check = Invoke-WebRequest -Uri "http://${wslAddress}:5000/health" -UseBasicParsing -TimeoutSec 10
    if ($check.StatusCode -ne 200) { throw "WSL MLflow health check failed." }
}

foreach ($address in $ListenAddress) {
    $listen = $address.IPAddressToString
    if ($address.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork -or
        $listen -eq "0.0.0.0" -or $listen -eq "127.0.0.1") {
        throw "Specify a concrete Windows LAN IPv4 address."
    }
    $ruleName = "SCD-MLflow-LAN-5000-$listen"
    $rule = Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
    if ($Remove) {
        # Ownership is marked by our dedicated firewall rule.
        if ($null -eq $rule) { throw "No managed MLflow rule found for $listen." }
        & netsh.exe interface portproxy delete v4tov4 "listenaddress=$listen" "listenport=5000"
        if ($LASTEXITCODE -ne 0) { throw "Port proxy removal failed." }
        Remove-NetFirewallRule -Name $ruleName
        Write-Host "Removed MLflow LAN access: ${listen}:5000"
        continue
    }
    $local = Get-NetIPAddress -AddressFamily IPv4 -IPAddress $listen -ErrorAction SilentlyContinue
    if ($null -eq $local) { throw "$listen is not assigned to this Windows PC." }
    $ipInterface = Get-NetIPInterface -InterfaceIndex $local[0].InterfaceIndex -AddressFamily IPv4
    if ($ipInterface.ConnectionState -ne "Connected") {
        throw "The adapter for $listen is disconnected. Connect it before exposing MLflow."
    }
    $existing = & netsh.exe interface portproxy show v4tov4
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect current port proxies." }
    $conflict = $existing | Where-Object {
        $_ -match ("^\s*" + [regex]::Escape($listen) + "\s+5000\s+") -or
        $_ -match '^\s*0\.0\.0\.0\s+5000\s+'
    }
    if ($conflict -and $null -eq $rule) {
        throw "An unmanaged TCP/5000 proxy already exists. Nothing was replaced."
    }
    & netsh.exe interface portproxy add v4tov4 "listenaddress=$listen" "listenport=5000" "connectaddress=$wslAddress" "connectport=5000"
    if ($LASTEXITCODE -ne 0) { throw "MLflow port proxy setup failed." }
    if ($null -eq $rule) {
        New-NetFirewallRule -Name $ruleName -DisplayName "SCD MLflow LAN TCP 5000 ($listen)" `
            -Direction Inbound -Action Allow -Protocol TCP -LocalPort 5000 `
            -LocalAddress $listen -RemoteAddress LocalSubnet -Profile Any | Out-Null
    }
    $response = Invoke-WebRequest -Uri "http://${listen}:5000/health" -UseBasicParsing -TimeoutSec 10
    if ($response.StatusCode -ne 200) { throw "LAN health check failed." }
    Write-Host "MLflow LAN endpoint ready: http://${listen}:5000 (LocalSubnet only)"
}
