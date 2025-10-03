@echo off
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8502" ^| findstr LISTENING') do set PID=%%a
if "%PID%"=="" (
  echo No process listening on port 8502 found.
  goto :eof
)
echo Found PID %PID% listening on :8502. Killing...
taskkill /PID %PID% /F
echo Done.
pause
