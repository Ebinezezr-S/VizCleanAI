@echo off
cd /d "%~dp0"
echo Building and starting docker-compose...
docker compose up --build
