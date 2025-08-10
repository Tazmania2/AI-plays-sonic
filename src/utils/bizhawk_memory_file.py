import os
import time
import threading
from pathlib import Path

class BizHawkMemoryReaderFile:
    def __init__(self, comm_dir=None, instance_id=0):
        # Use instance-specific communication directory
        if comm_dir is None:
            comm_dir = os.path.join(os.getcwd(), f"bizhawk_comm_{instance_id}")
        
        self.comm_dir = Path(comm_dir)
        self.request_file = self.comm_dir / "request.txt"
        self.response_file = self.comm_dir / "response.txt"
        self.status_file = self.comm_dir / "status.txt"
        self.ready = False
        self.instance_id = instance_id
        
        # Ensure communication directory exists
        self.comm_dir.mkdir(parents=True, exist_ok=True)
        print(f"[BizHawk-{instance_id}] Using communication directory: {self.comm_dir}")

    def connect(self, max_retries=30, retry_delay=1):
        """Wait for the Lua script to be ready."""
        print(f"[BizHawk-{self.instance_id}] Waiting for file-based bridge to be ready...")
        
        for attempt in range(max_retries):
            if os.path.exists(self.status_file):
                try:
                    with open(self.status_file, 'r') as f:
                        status = f.read().strip()
                    if status == "READY":
                        self.ready = True
                        print(f"[BizHawk-{self.instance_id}] File-based bridge is ready!")
                        return True
                except:
                    pass
            
            print(f"[BizHawk-{self.instance_id}] Waiting for bridge... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        
        print(f"[BizHawk-{self.instance_id}] File-based bridge not ready after {max_retries} attempts")
        return False

    def read_memory(self, address, size):
        """Read memory using file-based communication."""
        if not self.ready:
            if not self.connect():
                raise RuntimeError("Cannot connect to BizHawk file-based bridge")
        
        # Ensure communication directory exists
        self.comm_dir.mkdir(parents=True, exist_ok=True)
        
        # Write request
        try:
            with open(self.request_file, 'w') as f:
                f.write(f"read {address:x} {size}\n")
            
            # Wait for response (with optimized timeout)
            timeout = 1.0  # Reduced to 1 second for faster training
            start_time = time.time()
            
            while not self.response_file.exists():
                if time.time() - start_time > timeout:
                    print(f"[BizHawk-{self.instance_id}] Timeout waiting for response after {timeout}s")
                    # Return zero bytes on timeout instead of crashing
                    return bytes([0] * size)
                time.sleep(0.005)  # 5ms polling for faster response
            
            # Read response
            with open(self.response_file, 'rb') as f:
                data = f.read()
            
            # Clean up response file
            try:
                self.response_file.unlink()
            except:
                pass
            
            if len(data) != size:
                print(f"[BizHawk-{self.instance_id}] Warning: Expected {size} bytes, got {len(data)}")
                # Pad with zeros if too short, truncate if too long
                if len(data) < size:
                    data = data + bytes([0] * (size - len(data)))
                else:
                    data = data[:size]
            
            return data
            
        except Exception as e:
            print(f"[BizHawk-{self.instance_id}] File-based memory read error: {e}")
            # Return zero bytes on error instead of crashing
            return bytes([0] * size)

    def _send_command(self, command):
        """Send a command to the Lua bridge and get response."""
        if not self.ready:
            if not self.connect():
                raise RuntimeError("Cannot connect to BizHawk file-based bridge")
        
        # Ensure communication directory exists
        self.comm_dir.mkdir(parents=True, exist_ok=True)
        
        # Write request
        try:
            with open(self.request_file, 'w') as f:
                f.write(command + "\n")
            
            # Wait for response
            timeout = 1.0
            start_time = time.time()
            
            while not self.response_file.exists():
                if time.time() - start_time > timeout:
                    print(f"[BizHawk-{self.instance_id}] Timeout waiting for command response after {timeout}s")
                    return None
                time.sleep(0.005)
            
            # Read response
            with open(self.response_file, 'r') as f:
                response = f.read().strip()
            
            # Clean up response file
            try:
                self.response_file.unlink()
            except:
                pass
            
            return response
            
        except Exception as e:
            print(f"[BizHawk-{self.instance_id}] Command send error: {e}")
            return None 