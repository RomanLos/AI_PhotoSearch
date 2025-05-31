@echo off
echo ===============================================
echo           AI-PhotoSearch Updater
echo ===============================================
echo.

cd /d "%~dp0"

:: Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/
    pause
    exit /b 1
)

:: Check if git is initialized
if not exist ".git" (
    echo Initializing Git repository...
    git init
    git remote add origin https://github.com/RomanLos/AI_PhotoSearch.git
    echo Git repository initialized
    echo.
)

echo Checking for updates...
git fetch origin main

:: Check if updates are available
git diff HEAD origin/main --quiet
if %errorlevel% equ 0 (
    echo Already up to date!
    echo.
    pause
    exit /b 0
)

echo Updates found! Downloading...
echo.

:: Backup user settings
if exist "last_used_paths.pkl" (
    echo Backing up user settings...
    copy "last_used_paths.pkl" "last_used_paths.pkl.backup" >nul
)

:: Update files
git reset --hard origin/main
if %errorlevel% neq 0 (
    echo Update failed!
    pause
    exit /b 1
)

:: Restore user settings
if exist "last_used_paths.pkl.backup" (
    echo Restoring user settings...
    copy "last_used_paths.pkl.backup" "last_used_paths.pkl" >nul
    del "last_used_paths.pkl.backup" >nul
)

echo.
echo Update completed successfully!
echo Recent changes:
git log --oneline -5

echo.
echo You can now run the application
pause