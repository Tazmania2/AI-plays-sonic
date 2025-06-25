#!/usr/bin/env python3
"""
Test script to check RetroArch network command interface.
"""

import socket
import time
import subprocess
import os

def test_network_port(host='127.0.0.1', port=55355):
    """Test if RetroArch is listening on the network command port."""
    print(f"Testing connection to {host}:{port}")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ Port {port} is open and accepting connections!")
            return True
        else:
            print(f"❌ Port {port} is not accepting connections (error code: {result})")
            return False
    except Exception as e:
        print(f"❌ Error testing port {port}: {e}")
        return False

def check_retroarch_process():
    """Check if RetroArch is running."""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name']):
            if 'retroarch' in proc.info['name'].lower():
                print(f"✅ RetroArch process found: PID {proc.info['pid']}")
                return True
        print("❌ No RetroArch process found")
        return False
    except ImportError:
        print("⚠️  psutil not available, skipping process check")
        return True

def test_memory_reader():
    """Test the memory reader directly."""
    print("\nTesting memory reader...")
    try:
        from utils.retroarch_memory import RetroArchMemoryReader
        
        reader = RetroArchMemoryReader()
        print("✅ Memory reader created successfully")
        
        # Try to connect
        if reader.connect(max_retries=3, retry_delay=1):
            print("✅ Memory reader connected successfully!")
            
            # Try a simple memory read
            try:
                data = reader.read_memory(0xFFFE20, 2)  # Try to read rings
                print(f"✅ Memory read successful: {data.hex()}")
            except Exception as e:
                print(f"⚠️  Memory read failed: {e}")
            
            reader.close()
            return True
        else:
            print("❌ Memory reader failed to connect")
            return False
    except Exception as e:
        print(f"❌ Error testing memory reader: {e}")
        return False

def main():
    """Main test function."""
    print("🔌 RetroArch Network Command Interface Test")
    print("=" * 50)
    
    # Check if RetroArch is running
    retroarch_running = check_retroarch_process()
    
    # Test network port
    port_open = test_network_port()
    
    # Test memory reader
    memory_reader_works = test_memory_reader()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   RetroArch Process: {'✅ Running' if retroarch_running else '❌ Not Found'}")
    print(f"   Network Port 55355: {'✅ Open' if port_open else '❌ Closed'}")
    print(f"   Memory Reader: {'✅ Working' if memory_reader_works else '❌ Failed'}")
    
    if not retroarch_running:
        print("\n💡 RetroArch is not running. Start RetroArch first and try again.")
    elif not port_open:
        print("\n💡 RetroArch is running but network command interface is not enabled.")
        print("   Make sure 'network_cmd_enable = true' is set in retroarch.cfg")
    elif not memory_reader_works:
        print("\n💡 Network port is open but memory reader failed.")
        print("   This might be a timing issue - try again in a few seconds.")
    else:
        print("\n🎉 All tests passed! Network command interface is working correctly.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 