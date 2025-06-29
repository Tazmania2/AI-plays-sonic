@echo off
cd /d "D:\AI tests\test_ai_training"

REM Create a custom config directory
if not exist "config" mkdir config

REM Launch BizHawk with custom config
"C:\Program Files (x86)\BizHawk-2.10-win-x64\EmuHawk.exe" --lua="D:\AI tests\emulator\input_player.lua" "D:\AI tests\roms\Sonic The Hedgehog (USA, Europe).md"

pause 