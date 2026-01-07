# DynV6 IPv6 Update Script for Windows
# Run this periodically (e.g., via Task Scheduler) to keep your IPv6 updated

$hostname = "ohi-host.dynv6.net"
$token = "hpFFBiSbLyGjbK9dxQCL_wEk7iy_Js"

# Get the current public IPv6 address (GUA - Global Unicast Address starting with 2)
$ipv6 = (Get-NetIPAddress -AddressFamily IPv6 | Where-Object { 
    $_.IPAddress -like "2*" -and 
    $_.PrefixOrigin -eq "RouterAdvertisement" -and
    $_.AddressState -eq "Preferred"
} | Select-Object -First 1).IPAddress

if (-not $ipv6) {
    Write-Host "No public IPv6 address found!"
    exit 1
}

Write-Host "Current IPv6: $ipv6"
Write-Host "Updating $hostname..."

# Update dynv6
$url = "https://dynv6.com/api/update?hostname=$hostname&token=$token&ipv6=$ipv6"
try {
    $response = Invoke-WebRequest -Uri $url -UseBasicParsing
    Write-Host "Response: $($response.Content)"
} catch {
    Write-Host "Error: $_"
    exit 1
}

Write-Host "Done!"
