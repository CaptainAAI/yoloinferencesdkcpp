<#
  Builds YoloHttp.dll (Release x64) and packs it as an SDK folder + zip.

  Usage (from this folder):
    powershell -ExecutionPolicy Bypass -File pack_sdk.ps1

  Output: .\YoloHttp-SDK\  and  .\YoloHttp-SDK.zip
  OpenCV and curl are found the same way as in NewSDKhttp\YoloHttp.Build.props.
#>
param(
    [string]$OutDir = (Join-Path $PSScriptRoot "YoloHttp-SDK")
)
$ErrorActionPreference = "Stop"

$root   = $PSScriptRoot
$sdkSrc = Join-Path $root "sdk"
$binOut = Join-Path $root "bin\x64\Release"

$opencv = if ($env:YOLOHTTP_OPENCV_ROOT) { $env:YOLOHTTP_OPENCV_ROOT } else { Join-Path $root "..\library\opencv\build" }
$curl   = if ($env:YOLOHTTP_CURL_ROOT) { $env:YOLOHTTP_CURL_ROOT }
          elseif ($env:VCPKG_ROOT) { Join-Path $env:VCPKG_ROOT "installed\x64-windows" }
          else { "D:\vcpkg\installed\x64-windows" }

# ----- 1. Build the DLL (Release x64) -----
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$msbuild = & $vswhere -latest -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
if (-not $msbuild) { throw "MSBuild not found" }
Write-Host "Building NewSDKhttp.sln (Release|x64)..."
& $msbuild (Join-Path $root "NewSDKhttp.sln") /p:Configuration=Release /p:Platform=x64 /m /v:minimal /nologo
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

# ----- 2. SDK folder -----
if (Test-Path $OutDir) { Remove-Item $OutDir -Recurse -Force }
foreach ($d in "include", "lib", "bin", "licenses") {
    New-Item -ItemType Directory -Force (Join-Path $OutDir $d) | Out-Null
}

Copy-Item (Join-Path $root "NewSDKhttp\include\YoloHttp.h")     (Join-Path $OutDir "include")
Copy-Item (Join-Path $opencv "include\opencv2")                 (Join-Path $OutDir "include") -Recurse

Copy-Item (Join-Path $binOut "YoloHttp.lib")                    (Join-Path $OutDir "lib")
Copy-Item (Join-Path $opencv "x64\vc16\lib\opencv_world500.lib") (Join-Path $OutDir "lib")

Copy-Item (Join-Path $binOut "YoloHttp.dll")                    (Join-Path $OutDir "bin")
Copy-Item (Join-Path $curl "bin\libcurl.dll")                   (Join-Path $OutDir "bin")
Copy-Item (Join-Path $curl "bin\z.dll")                         (Join-Path $OutDir "bin")
Copy-Item (Join-Path $opencv "x64\vc16\bin\opencv_world500.dll") (Join-Path $OutDir "bin")

Copy-Item (Join-Path $sdkSrc "YoloHttp.props")                  $OutDir
# The repository README is the SDK's full guide. The Jetson side (server, dashboard,
# image server) is deployed from its own repositories, not from this package.
Copy-Item (Join-Path $root "README.md")                         $OutDir

$lic = Join-Path $OutDir "licenses"
Copy-Item (Join-Path $sdkSrc "THIRD_PARTY_NOTICES.txt")         $lic
Copy-Item (Join-Path $curl "share\curl\copyright")              (Join-Path $lic "curl-COPYING.txt")
Copy-Item (Join-Path $curl "share\zlib\copyright")              (Join-Path $lic "zlib-LICENSE.txt")
Copy-Item (Join-Path $curl "share\rapidjson\copyright")         (Join-Path $lic "rapidjson-LICENSE.txt")
Copy-Item (Join-Path $opencv "LICENSE")                         (Join-Path $lic "OpenCV-LICENSE.txt")

# ----- 3. Zip -----
$zip = "$OutDir.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $OutDir "*") -DestinationPath $zip

$size = (Get-ChildItem $OutDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("SDK : {0} ({1:N1} MB)" -f (Resolve-Path $OutDir), $size)
Write-Host ("Zip : {0} ({1:N1} MB)" -f (Resolve-Path $zip), ((Get-Item $zip).Length / 1MB))
