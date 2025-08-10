#!/usr/bin/env python3
"""
Process Cleanup Script for Sonic AI Training

This script ensures all training processes, emulators, and related applications
are properly terminated to prevent resource conflicts and hanging processes.
"""

import subprocess
import psutil
import time
import sys
from pathlib import Path

def kill_processes_by_name(process_names):
    """Kill all processes with the given names."""
    killed_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            proc_name = proc.info['name'].lower()
            cmdline = ' '.join(proc.info['cmdline'] or []).lower()
            
            for target_name in process_names:
                if target_name.lower() in proc_name or target_name.lower() in cmdline:
                    print(f"🔄 Terminating {proc_name} (PID: {proc.info['pid']})")
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)  # Wait for graceful termination
                        print(f"✅ Successfully terminated {proc_name}")
                        killed_count += 1
                    except psutil.TimeoutExpired:
                        print(f"⚠️  Force killing {proc_name}")
                        proc.kill()
                        killed_count += 1
                    except psutil.NoSuchProcess:
                        pass  # Process already terminated
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return killed_count

def cleanup_training_processes():
    """Clean up all AI training related processes."""
    print("🧹 Cleaning up AI training processes...")
    
    # Processes to terminate
    training_processes = [
        'python', 'train_sonic.py', 'play_sonic.py', 'test_sonic.py',
        'bizhawk', 'retroarch', 'emulator', 'sonic'
    ]
    
    killed = kill_processes_by_name(training_processes)
    
    if killed > 0:
        print(f"✅ Terminated {killed} training processes")
    else:
        print("✅ No training processes found")

def cleanup_emulator_processes():
    """Clean up all emulator processes."""
    print("🎮 Cleaning up emulator processes...")
    
    emulator_processes = [
        'bizhawk', 'retroarch', 'emulator', 'lua'
    ]
    
    killed = kill_processes_by_name(emulator_processes)
    
    if killed > 0:
        print(f"✅ Terminated {killed} emulator processes")
    else:
        print("✅ No emulator processes found")

def cleanup_file_bridges():
    """Clean up any file bridge communication files."""
    print("📁 Cleaning up file bridge communication...")
    
    bridge_files = [
        'bizhawk_comm/status.txt',
        'bizhawk_comm/memory_dump.bin',
        'bizhawk_comm/command.txt'
    ]
    
    for file_path in bridge_files:
        try:
            if Path(file_path).exists():
                Path(file_path).unlink()
                print(f"✅ Removed {file_path}")
        except Exception as e:
            print(f"⚠️  Could not remove {file_path}: {e}")

def check_system_resources():
    """Check system resources after cleanup."""
    print("\n📊 System Resource Check:")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"Memory Usage: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
    
    # GPU check (if available)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            print(f"GPU Usage: {gpu_info[0]}% | GPU Memory: {gpu_info[1]}MB / {gpu_info[2]}MB")
    except:
        print("GPU: Not available or nvidia-smi not found")

def main():
    """Main cleanup process."""
    print("🚀 Sonic AI Training Process Cleanup")
    print("=" * 50)
    
    # Perform cleanup
    cleanup_training_processes()
    cleanup_emulator_processes()
    cleanup_file_bridges()
    
    # Wait a moment for processes to fully terminate
    print("\n⏳ Waiting for processes to fully terminate...")
    time.sleep(2)
    
    # Check system resources
    check_system_resources()
    
    print("\n✅ Cleanup complete! System is ready for new training sessions.")
    print("\n💡 Tips:")
    print("- Always run this script before starting new training sessions")
    print("- Use Ctrl+C to stop training scripts gracefully")
    print("- Check Task Manager if you suspect hanging processes")

if __name__ == "__main__":
    main() 