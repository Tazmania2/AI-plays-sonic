#!/usr/bin/env python3
"""
Test script to check RetroArch network command interface with longer delays.
"""

import socket
import time
import subprocess
import os

def test_connection_with_delay(host='127.0.0.1', port=55355, max_delay=30):
    """Test connection with increasing delays."""
    print(f"Testing connection to {host}:{port} with delays up to {max_delay} seconds...")
    
    for delay in range(0, max_delay + 1, 5):
        print(f"\nTrying after {delay} second delay...")
        if delay > 0:
            time.sleep(delay)
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"✅ SUCCESS! Port {port} is open after {delay} seconds!")
                return True
            else:
                print(f"❌ Port {port} still closed (error code: {result})")
        except Exception as e:
            print(f"❌ Error testing port {port}: {e}")
    
    print(f"❌ Port {port} never became available after {max_delay} seconds")
    return False

def test_memory_reader_with_delay(max_delay=30):
    """Test the memory reader with delays."""
    print(f"\nTesting memory reader with delays up to {max_delay} seconds...")
    
    for delay in range(0, max_delay + 1, 5):
        print(f"\nTrying memory reader after {delay} second delay...")
        if delay > 0:
            time.sleep(delay)
        
        try:
            from utils.retroarch_memory import RetroArchMemoryReader
            
            reader = RetroArchMemoryReader()
            print("✅ Memory reader created successfully")
            
            # Try to connect
            if reader.connect(max_retries=2, retry_delay=1):
                print(f"✅ SUCCESS! Memory reader connected after {delay} seconds!")
                
                # Try a simple memory read
                try:
                    data = reader.read_memory(0xFFFE20, 2)  # Try to read rings
                    print(f"✅ Memory read successful: {data.hex()}")
                except Exception as e:
                    print(f"⚠️  Memory read failed: {e}")
                
                reader.close()
                return True
            else:
                print(f"❌ Memory reader still failed to connect after {delay} seconds")
        except Exception as e:
            print(f"❌ Error testing memory reader: {e}")
    
    print(f"❌ Memory reader never connected after {max_delay} seconds")
    return False

def main():
    """Main test function."""
    print("🔌 RetroArch Network Command Interface Test (with delays)")
    print("=" * 60)
    
    # Test basic port connection with delays
    port_works = test_connection_with_delay()
    
    # Test memory reader with delays
    memory_reader_works = test_memory_reader_with_delay()
    
    print("\n" + "=" * 60)
    print("📊 Final Results:")
    print(f"   Network Port 55355: {'✅ Working' if port_works else '❌ Failed'}")
    print(f"   Memory Reader: {'✅ Working' if memory_reader_works else '❌ Failed'}")
    
    if not port_works:
        print("\n💡 Network command interface never became available.")
        print("   This suggests RetroArch is not using the correct config file")
        print("   or the network command interface is not enabled.")
    elif not memory_reader_works:
        print("\n💡 Port is open but memory reader failed.")
        print("   This might be a protocol issue.")
    else:
        print("\n🎉 All tests passed! Network command interface is working correctly.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 