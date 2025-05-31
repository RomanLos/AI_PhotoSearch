@echo off
echo ===============================================
echo           AI-PhotoSearch Updater
echo ===============================================
echo.

cd /d "%~dp0"

git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/
    pause
    exit /b 1
)

if not exist ".git" (
    echo 🔄 Initializing Git repository...
    git init
    git remote add origin https://github.com/RomanLos/AI_PhotoSearch.git
    echo ✅ Git repository initialized
    echo.
)

echo 🔍 Checking for updates...
git fetch origin main

git diff HEAD origin/main --quiet
if %errorlevel% equ 0 (
    echo ✅ Already up to date!
    echo.
    pause
    exit /b 0
)

echo 📦 Updates found! Downloading...

if exist "last_used_paths.pkl" (
    echo 💾 Backing up user settings...
    copy "last_used_paths.pkl" "last_used_paths.pkl.backup" >nul
)

git reset --hard origin/main

if exist "last_used_paths.pkl.backup" (
    echo 🔄 Restoring user settings...
    copy "last_used_paths.pkl.backup" "last_used_paths.pkl" >nul
    del "last_used_paths.pkl.backup" >nul
)

echo ✅ Update completed successfully!
git log --oneline -5
pause