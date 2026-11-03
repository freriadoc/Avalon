@echo off
rem Avalon on Windows: Qwen3.8-Flash-Next with MTP, 256k context, OpenAI-compatible API on http://%HOST%:%PORT%
rem (default http://127.0.0.1:8080). Settings (environment): MODEL_DIR, HOST, PORT; LLAMA_API_KEY is read by
rem llama-server. Extra arguments go to llama-server (e.g. -c 65536 for a smaller context).
setlocal

if not defined MODEL_DIR set "MODEL_DIR=%USERPROFILE%\Avalon-models\Qwen3.8-Flash-Next-Avalon-GGUF"
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8080"
set "SERVER=%~dp0..\build\bin\llama-server.exe"

if not defined HIP_PATH (
    echo error: HIP_PATH is not set, install the AMD HIP SDK
    goto :fail
)
if not exist "%SERVER%" (
    echo error: %SERVER% not found, build it first ^(docs\windows-port.md^)
    goto :fail
)
if not exist "%MODEL_DIR%\Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf" (
    echo error: the model is not in %MODEL_DIR% ^(set MODEL_DIR^)
    goto :fail
)

rem the HIP runtime DLLs
set "PATH=%HIP_PATH%bin;%PATH%"
rem pinning would register the whole ~89 GiB model with the driver: not tested on Windows, risky with less RAM
if not defined GGML_CUDA_MOE_PIN set "GGML_CUDA_MOE_PIN=0"

cd /d "%MODEL_DIR%"
"%SERVER%" -m Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf ^
  -c 262144 -b 9536 -ub 9536 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1 ^
  -md MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf --spec-type draft-mtp ^
  --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0 ^
  --host %HOST% --port %PORT% %*
if errorlevel 1 goto :fail
exit /b 0

:fail
pause
exit /b 1
