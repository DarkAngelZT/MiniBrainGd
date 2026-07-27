@echo off
setlocal
cd /d "%~dp0"

:: Usage: build_tests.bat LLVM_BIN_PATH [jobs]
:: Example: build_tests.bat C:\Files\programs\llvm-mingw-ucrt-x86_64\bin 8

if "%~1"=="" (
    echo ERROR: Please provide the llvm-mingw bin path.
    echo Usage: build_tests.bat LLVM_BIN_PATH [jobs]
    exit /b 2
)

set "LLVM_BIN=%~1"
set "JOBS=%NUMBER_OF_PROCESSORS%"
if not "%~2"=="" set "JOBS=%~2"
set "PATH=%LLVM_BIN%;%PATH%"

echo ========== Building MiniBrainGd C++ tests ==========
scons -f SConstruct.tests llvm_bin="%LLVM_BIN%" -j%JOBS%
if errorlevel 1 exit /b %ERRORLEVEL%

echo ========== Running MiniBrainGd C++ tests ==========
tests\bin\minibrain_gd_tests.exe
exit /b %ERRORLEVEL%
