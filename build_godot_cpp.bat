@echo off
cls
cd godot-cpp
setlocal enabledelayedexpansion

:: Usage: build_extension.bat LLVM_BIN_PATH [target]
:: Example 1 (Dev): build_extension.bat D:\llvm-mingw\bin
:: Example 2 (Release): build_extension.bat D:\llvm-mingw\bin template_release

if "%~1"=="" (
    echo ERROR: Please input llvm-mingw bin path!
    echo Example: build_extension.bat D:\llvm-mingw\bin
    cd ..
    pause
    exit /b
)
set "LLVM_BIN=%~1"

set "TARGET=editor"
if not "%~2"=="" set "TARGET=%~2"

set "PATH=!LLVM_BIN!;%PATH%"

echo ========== Building GDExtension ==========
echo LLVM Path: !LLVM_BIN!
echo Target: !TARGET!
echo ==========================================

scons platform=windows use_llvm=yes use_mingw=yes target=!TARGET! -j%NUMBER_OF_PROCESSORS%

if %errorlevel% equ 0 (
    echo Build Success! Output in bin folder.
) else (
    echo Build Failed!
)

cd ..
pause
endlocal