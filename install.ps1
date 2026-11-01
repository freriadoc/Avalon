# Avalon installer (Windows, AMD HIP SDK): checks the machine, installs the build tools it lacks, builds llama-server
# for the detected GPU, downloads the model with its MTP draft head, verifies every file against its SHA-256 on Hugging
# Face and installs an Avalon launcher with a desktop shortcut.
#
#   double-click install.cmd in a clone, or in PowerShell:
#   irm https://raw.githubusercontent.com/freriadoc/Avalon/main/install.ps1 | iex
#
# Settings (environment): AVALON_DIR (source tree, default %USERPROFILE%\Avalon), MODEL_DIR (default
# %USERPROFILE%\Avalon-models\Qwen3.8-Flash-Next-Avalon-GGUF), AMDGPU_TARGETS (skip the detection), SKIP_MODEL=1 (build
# only), FORCE=1 (go on past a failed check).
#
# What it needs and installs (winget, each asks for administrator rights once): Git, Python 3, Visual Studio 2022 Build
# Tools with the MSVC 14.44 toolset, CMake and Ninja. The AMD HIP SDK is not on winget: the installer opens its download
# page and stops; run it again once the SDK is installed. Build notes: docs of the Windows port by ppkrol
# (https://github.com/ppkrol/Avalon_windows, docs/windows-port.md).

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

$RepoUrl   = 'https://github.com/freriadoc/Avalon'
$HfRepo    = 'freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF'
$AvalonDir = if ($env:AVALON_DIR) { $env:AVALON_DIR } else { Join-Path $env:USERPROFILE 'Avalon' }
$ModelDir  = if ($env:MODEL_DIR)  { $env:MODEL_DIR }  else { Join-Path $env:USERPROFILE 'Avalon-models\Qwen3.8-Flash-Next-Avalon-GGUF' }
$Force     = $env:FORCE -eq '1'
$SkipModel = $env:SKIP_MODEL -eq '1'
# the HIP SDK's clang 21 rejects the <cmath> of MSVC 14.51 (VS 18): 14.44 (VS 2022 17.14) is the toolset that builds
$MsvcVer   = '14.44'
$MsvcComp  = 'Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64'

function Say($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "warning: $m" -ForegroundColor Yellow }
function Fail($m) {
    Write-Host "error: $m" -ForegroundColor Red
    if ($Force) { Warn 'FORCE=1: going on'; return }
    throw "Avalon installer stopped: $m"
}
# a tool installed during this run is on the machine's PATH, not yet on this process's
function Update-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}
function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Winget-Install($id, $extra) {
    if (-not (Have 'winget')) { Fail "winget not found: install '$id' by hand (or App Installer from the Microsoft Store)"; return }
    Say "Installing $id (winget)"
    $wargs = @('install', '--id', $id, '-e', '--accept-package-agreements', '--accept-source-agreements')
    if ($extra) { $wargs += $extra }
    & winget @wargs
    Update-Path
}

try {
# ---------------------------------------------------------------------------------------------------------------------
Say 'Checking the machine'

if (-not [Environment]::Is64BitOperatingSystem) { Fail 'a 64-bit Windows is needed' }

# AVX2: the experts computed on the CPU need it (PF_AVX2_INSTRUCTIONS_AVAILABLE = 40)
Add-Type -Namespace Avalon -Name Cpu -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint f);'
if (-not [Avalon.Cpu]::IsProcessorFeaturePresent(40)) { Fail 'the CPU has no AVX2: the experts computed on the CPU need it' }

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
Write-Host ("CPU {0} ({1} cores)" -f $cpu.Name.Trim(), $cpu.NumberOfCores)

# RAM: the model (~89 GiB) is memory-mapped; what does not fit is read again from the disk. Measured on Linux with the
# server capped at 28 GB: generation -2 to -6 %, prefill of a new 19k prompt -31 %. Windows with 64 GB (the port): the
# first requests after a start are slower while the file cache fills.
$ramGiB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Host "RAM $ramGiB GiB"
if ($ramGiB -lt 28) { Fail "$ramGiB GiB of RAM: Avalon is measured with 32 GB or more (the model is memory-mapped, ~89 GiB)" }

# GPU: RDNA3 (gfx1100 measured, the Windows port ran on a 7900 XTX) or RDNA4 (reported working on Linux by a user, a
# Radeon AI PRO R9700, not measured here); the exact target comes from hipInfo later
$gpus = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Radeon' } | ForEach-Object { $_.Name })
if ($gpus.Count -eq 0) { Fail 'no AMD Radeon GPU found' } else { Write-Host ("GPU " + ($gpus -join ', ')) }
if (-not ($gpus -match 'RX 7[0-9]{3}|RX 9[0-9]{3}|R9[0-9]{3}')) { Warn 'no Radeon RX 7000 / RX 9000 / AI PRO R9000 (RDNA3 / RDNA4) found by name: Avalon''s kernels target gfx1100 - gfx1102 and gfx1200 / gfx1201' }

# VRAM: the driver's 64-bit size in the display class keys (Win32_VideoController.AdapterRAM stops at 4 GB)
$vramMiB = 0
Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' -ErrorAction SilentlyContinue |
    ForEach-Object {
        $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
        if ($p -and $p.DriverDesc -match 'Radeon' -and $p.'HardwareInformation.qwMemorySize') {
            $v = [math]::Floor([int64]$p.'HardwareInformation.qwMemorySize' / 1MB)
            if ($v -gt $vramMiB) { $vramMiB = $v }
        }
    }
if ($vramMiB -gt 0) {
    Write-Host ("VRAM {0} GB" -f [math]::Floor($vramMiB * 1MB / 1e9))
    if ($vramMiB -lt 19000) { Warn 'less than 20 GB of VRAM: the 256k context with MTP may not fit; try avalon-server -c 65536' }
}

if (-not $SkipModel) {
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    $have = (Get-ChildItem $ModelDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    $drive = Get-PSDrive -Name (Split-Path -Qualifier (Resolve-Path $ModelDir).Path).TrimEnd(':')
    $freeGiB = [math]::Floor(($drive.Free + [int64]$have) / 1GB)
    if ($freeGiB -lt 98) { Fail "$freeGiB GiB free for ${ModelDir}: the model needs ~97 GiB (set MODEL_DIR to another drive, an NVMe one)" }
}

# ---------------------------------------------------------------------------------------------------------------------
Say 'Checking the build tools'

Update-Path
if (-not (Have 'git')) { Winget-Install 'Git.Git' }
if (-not (Have 'git')) { Fail 'git still not found (open a new terminal and run the installer again)' }

# Python 3.9 or newer: the py launcher, else python on the PATH
function Find-Python {
    foreach ($c in @(@('py', '-3'), @('python'))) {
        if (-not (Have $c[0])) { continue }
        $pre = @($c | Select-Object -Skip 1)
        try { $ok = & $c[0] @pre -c 'import sys; print(sys.version_info >= (3, 9))' 2>$null } catch { $ok = $null }
        if ($ok -eq 'True') { return ,$c }
    }
    return $null
}
$python = Find-Python
if (-not $python) {
    Winget-Install 'Python.Python.3.12'
    $python = Find-Python
    if (-not $python) { Fail 'Python 3 still not found (open a new terminal and run the installer again)' }
}

# Visual Studio Build Tools with the MSVC 14.44 toolset, CMake and Ninja (the VC CMake component brings both). In VS 2022
# 17.14, 14.44 is the default toolset; VS 18 has it as an optional versioned component (what the Windows port used)
$vsInstaller = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer'
$vswhere = Join-Path $vsInstaller 'vswhere.exe'
function Get-VSList {
    if (-not (Test-Path $vswhere)) { return @() }
    $json = & $vswhere -products * -format json | Out-String
    if (-not $json.Trim()) { return @() }
    # Windows PowerShell 5.1 outputs a JSON array as one object: enumerate it
    $o = ConvertFrom-Json $json
    return @($o | ForEach-Object { $_ })
}
function Find-VS {
    foreach ($i in Get-VSList) {
        $p = $i.installationPath
        $msvc = Get-ChildItem (Join-Path $p 'VC\Tools\MSVC') -Directory -Filter "$MsvcVer.*" -ErrorAction SilentlyContinue
        $cm = Test-Path (Join-Path $p 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe')
        if ($msvc -and $cm) { return $p }
    }
    return $null
}
$vs = Find-VS
if (-not $vs) {
    $list = Get-VSList
    $common = '--add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.CMake.Project --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended'
    if ($list.Count -eq 0) {
        Winget-Install 'Microsoft.VisualStudio.2022.BuildTools' @('--override', "--quiet --wait --norestart $common --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64")
    } else {
        $i = $list[0]
        $add = if ([int]($i.installationVersion.Split('.')[0]) -ge 18) { "--add $MsvcComp" } else { '--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64' }
        Say "Adding the MSVC $MsvcVer toolset, CMake and Ninja to $($i.installationPath) (administrator rights asked)"
        Start-Process -FilePath (Join-Path $vsInstaller 'setup.exe') -Verb RunAs -Wait `
            -ArgumentList ("modify --installPath `"$($i.installationPath)`" $common $add --quiet --norestart")
    }
    $vs = Find-VS
    if (-not $vs) { Fail "no Visual Studio with the MSVC $MsvcVer toolset and CMake after the install (Visual Studio Installer: add 'MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.44-17.14)' and 'C++ CMake tools for Windows')" }
}
Write-Host "Visual Studio: $vs (MSVC $MsvcVer)"
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'

# AMD HIP SDK (sets HIP_PATH, machine-wide)
$hip = $env:HIP_PATH
if (-not $hip) { $hip = [Environment]::GetEnvironmentVariable('HIP_PATH', 'Machine') }
if (-not $hip -or -not (Test-Path (Join-Path $hip 'bin\clang.exe'))) {
    Start-Process 'https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html'
    Fail 'the AMD HIP SDK is not installed: download and install it (HIP SDK 7.2 for Windows, the page has just opened), then run this installer again'
    return
}
$hip = $hip.TrimEnd('\')
Write-Host "HIP SDK: $hip"
if ($hip -notmatch '\\7\.\d+$') { Warn "HIP SDK ${hip}: the Windows port was built with HIP SDK 7.2" }

# GPU target from the HIP runtime
$targets = $env:AMDGPU_TARGETS
if (-not $targets) {
    $hipInfo = Join-Path $hip 'bin\hipInfo.exe'
    $arch = @()
    if (Test-Path $hipInfo) {
        $env:Path = "$hip\bin;$env:Path"
        $arch = @((& $hipInfo 2>$null) | Select-String -Pattern 'gfx[0-9a-f]+' -AllMatches | ForEach-Object { $_.Matches.Value } | Sort-Object -Unique)
    }
    foreach ($g in @('gfx1100', 'gfx1101', 'gfx1102', 'gfx1201', 'gfx1200')) { if ($arch -contains $g) { $targets = $g; break } }
    if (-not $targets) {
        Fail ("no RDNA3 / RDNA4 discrete GPU found by hipInfo (it reports: {0}). Avalon's kernels target gfx1100 - gfx1102 and gfx1200 / gfx1201" -f ($(if ($arch) { $arch -join ' ' } else { 'nothing' })))
        $targets = if ($arch) { $arch[0] } else { 'gfx1100' }
    }
}
if ($targets -eq 'gfx1100') { Write-Host "GPU target $targets (RX 7900 XT / XTX: the measured target)" }
elseif ($targets -match '^gfx120[01]$') { Warn "GPU target $targets (RDNA4): reported working on Linux by a user (R9700), not measured here nor on Windows; some prompt kernels are RDNA3-only" }
else { Warn "GPU target ${targets}: should work, not measured (Avalon is tuned on a gfx1100)" }

# ---------------------------------------------------------------------------------------------------------------------
Say 'Getting the source'

$here = if ($PSScriptRoot) { $PSScriptRoot } else { $null }
if ($here -and (Test-Path (Join-Path $here 'ggml\src\ggml-cuda\moe-cache.cu'))) {
    $AvalonDir = $here
    Write-Host "using the clone in $AvalonDir"
} elseif (Test-Path (Join-Path $AvalonDir '.git')) {
    Write-Host "updating $AvalonDir"
    & git -C $AvalonDir pull --ff-only
    if ($LASTEXITCODE -ne 0) { Fail "git pull failed in $AvalonDir" }
} else {
    & git clone --depth 1 $RepoUrl $AvalonDir
    if ($LASTEXITCODE -ne 0) { Fail "git clone of $RepoUrl failed" }
}

# ---------------------------------------------------------------------------------------------------------------------
Say "Building llama-server for $targets (15 - 40 minutes)"

$build  = Join-Path $AvalonDir 'build'
$hipFwd = $hip -replace '\\', '/'
# static libraries: ggml_moe_cache is a variable shared by the ggml DLLs (dllexport in every backend); no tests (Linux
# scripts). OpenMP comes from the MSVC toolset (libomp), the fast path of the CPU experts needs it.
$cmd = @"
@echo off
call "$vcvars" -vcvars_ver=$MsvcVer >nul || exit /b 1
set "PATH=$hip\bin;%PATH%"
cmake -S "$AvalonDir" -B "$build" -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=$targets -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_PREFIX_PATH="$hipFwd" ^
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF > "$AvalonDir\build-configure.log" 2>&1 || exit /b 2
findstr /c:"Found OpenMP" "$AvalonDir\build-configure.log" >nul || echo warning: OpenMP not found, the CPU experts will be slower
cmake --build "$build" --target llama-server > "$AvalonDir\build.log" 2>&1 || exit /b 3
"@
$cmdFile = Join-Path $env:TEMP 'avalon-build.cmd'
Set-Content -Path $cmdFile -Value $cmd -Encoding ASCII
& cmd.exe /c $cmdFile
switch ($LASTEXITCODE) {
    0 { }
    1 { Fail "vcvars64.bat -vcvars_ver=$MsvcVer failed: the MSVC $MsvcVer toolset is missing from $vs" }
    2 { Get-Content "$AvalonDir\build-configure.log" -Tail 25; Fail "cmake failed (log: $AvalonDir\build-configure.log)" }
    default { Get-Content "$AvalonDir\build.log" -Tail 40; Fail "build failed (log: $AvalonDir\build.log)" }
}
$server = Join-Path $build 'bin\llama-server.exe'
if (-not (Test-Path $server)) { Fail "$server not found after the build" }
Write-Host "built $server"

# ---------------------------------------------------------------------------------------------------------------------
if (-not $SkipModel) {
    Say 'Downloading the model (~89 GiB, resumable: run the installer again if it stops)'

    $venv = Join-Path $AvalonDir '.venv'
    $vpy  = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path $vpy)) {
        $pre = @($python | Select-Object -Skip 1)
        & $python[0] @pre -m venv $venv
        if ($LASTEXITCODE -ne 0) { Fail 'python -m venv failed' }
    }
    & $vpy -m pip install -q --upgrade 'huggingface_hub>=0.34'
    & (Join-Path $venv 'Scripts\hf.exe') download $HfRepo --local-dir $ModelDir --exclude '*.jpg' --exclude '*.md'
    if ($LASTEXITCODE -ne 0) { Fail 'the download stopped: run the installer again to resume it' }

    Say 'Verifying the SHA-256 of every file'
    $verify = @'
import hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
repo, root = sys.argv[1], sys.argv[2]
tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"))
files = [(f["path"], f["lfs"]["oid"], f["size"]) for f in tree if f["type"] == "file" and f.get("lfs") and not f["path"].endswith(".jpg")]
done_path = os.path.join(root, ".avalon-verified")
done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
def check(item):
    path, oid, size = item
    p = os.path.join(root, path)
    if not os.path.exists(p) or os.path.getsize(p) != size:
        return path, "missing or incomplete"
    key = f"{path}:{oid}:{int(os.path.getmtime(p))}"
    if key in done:
        return path, key
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return path, key if h.hexdigest() == oid else "SHA-256 mismatch"
bad = []
with ThreadPoolExecutor(max(1, min(8, os.cpu_count() or 1))) as ex:
    for path, r in ex.map(check, files):
        if r.count(":") >= 2:
            done.add(r)
        else:
            bad.append(path); print(f"  {path}: {r}")
open(done_path, "w").write("\n".join(sorted(done)) + "\n")
if bad:
    for path in bad:
        try: os.remove(os.path.join(root, path))
        except FileNotFoundError: pass
    sys.exit(f"{len(bad)} file(s) removed: run the installer again to download them")
print(f"  {len(files)} files OK")
'@
    $verifyFile = Join-Path $env:TEMP 'avalon-verify.py'
    Set-Content -Path $verifyFile -Value $verify -Encoding UTF8
    & $vpy $verifyFile $HfRepo $ModelDir
    if ($LASTEXITCODE -ne 0) { Fail 'some model files were bad and have been removed: run the installer again to download them' }
}

# ---------------------------------------------------------------------------------------------------------------------
$launcher = Join-Path $AvalonDir 'avalon-server.cmd'
Say "Installing the launcher $launcher"

$cmd = @"
@echo off
rem Avalon: Qwen3.8-Flash-Next with MTP, 256k context, OpenAI-compatible API and chat page on http://%HOST%:%PORT%
rem (default http://127.0.0.1:8080). Settings (environment): HOST, PORT; LLAMA_API_KEY is read by llama-server.
rem Extra arguments go to llama-server (e.g. -c 65536 for a smaller context). Written by install.ps1.
setlocal
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8080"
rem the HIP runtime DLLs
set "PATH=$hip\bin;%PATH%"
rem pinning would register the whole ~89 GiB model with the driver: not tested on Windows, risky with less RAM
if not defined GGML_CUDA_MOE_PIN set "GGML_CUDA_MOE_PIN=0"
cd /d "$ModelDir"
echo Avalon starts: the chat page and the API will be on http://%HOST%:%PORT% once the model is loaded (about a minute)
"$server" -m Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf ^
  -c 262144 -b 9536 -ub 9536 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1 ^
  -md MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf --spec-type draft-mtp ^
  --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0 ^
  --host %HOST% --port %PORT% %*
if errorlevel 1 pause
"@
Set-Content -Path $launcher -Value $cmd -Encoding ASCII

$desktop = [Environment]::GetFolderPath('Desktop')
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path $desktop 'Avalon server.lnk'))
$lnk.TargetPath = $launcher
$lnk.WorkingDirectory = $AvalonDir
$lnk.Description = 'Avalon: Qwen3.8-Flash-Next on http://127.0.0.1:8080'
$lnk.Save()

Say "Done. Start it with the 'Avalon server' shortcut on the desktop (or $launcher), then open http://127.0.0.1:8080"
} catch {
    Write-Host $_ -ForegroundColor Red
    # run from a file (install.cmd): an exit code; run by irm | iex: no exit, it would close the user's window
    if ($PSCommandPath) { exit 1 }
}
