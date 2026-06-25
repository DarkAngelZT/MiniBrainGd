@echo off
cls
cd /d "%~dp0"
setlocal enabledelayedexpansion

:: Usage: build_all.bat LLVM_BIN_PATH [target] [jobs]
:: Example 1 (Dev): build_all.bat D:\llvm-mingw\bin editor
:: Example 2 (Release): build_all.bat D:\llvm-mingw\bin template_release 8

if "%~1"=="" (
    echo ERROR: Please input llvm-mingw bin path!
    echo Usage: build_all.bat LLVM_BIN_PATH [target] [jobs]
    goto end
)
set "LLVM_BIN=%~1"

set "TARGET=editor"
if not "%~2"=="" set "TARGET=%~2"

set "JOBS=%NUMBER_OF_PROCESSORS%"
if not "%~3"=="" set "JOBS=%~3"

set "PATH=%LLVM_BIN%;%PATH%"

echo ========== Building MiniBrainGd ==========
echo LLVM Path: %LLVM_BIN%
echo Target: %TARGET%
echo Jobs: %JOBS%
echo ==========================================

scons platform=windows use_llvm=yes use_mingw=yes target=%TARGET% -j%JOBS%

if %ERRORLEVEL% equ 0 (
    echo.
    echo Build Success! Output in bin folder.
) else (
    echo.
    echo Build Failed! Check SCons output above.
)

:end
pause
endlocal
