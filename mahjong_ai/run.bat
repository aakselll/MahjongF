@echo off
chcp 65001 > nul

REM ─── Configuration ──────────────────────────────────────────────
set "User="
set "ProjectDir="
set "PythonDir="
set "MitmDump=%PythonDir%\Scripts\mitmdump.exe"

REM ─── Aller dans le dossier du projet ────────────────────────────
cd /d "%ProjectDir%"
if errorlevel 1 (
    echo [ERREUR] Impossible d'acceder a %ProjectDir%
    pause
    exit /b 1
)

REM ─── Verifier que mitmdump existe ───────────────────────────────
if not exist "%MitmDump%" (
    echo [ERREUR] mitmdump introuvable : %MitmDump%
    echo         Verifie la variable User="%User%"
    pause
    exit /b 1
)

REM ─── Lancement de mitmdump avec l'addon ─────────────────────────
echo [INFO] Lancement de mitmdump (User=%User%)...
echo [INFO] Projet : %ProjectDir%
echo [INFO] Python : %PythonDir%
echo.

"%MitmDump%" -q -s sniffer_suphx.py --mode socks5 --listen-port 7878

pause