# DEPRECATED: Socket-based memory reader - replaced by file-based bridge
# This file is kept for reference but should not be used in new code

# import socket
# import time

# class BizHawkMemoryReader:
#     def __init__(self, host='127.0.0.1', port=55555):
#         self.host = host
#         self.port = port
#         self.connected = False

#     def connect(self, max_retries=10, retry_delay=1):
#         # Try to connect to the BizHawk Lua bridge.
#         for attempt in range(max_retries):
#             try:
#                 print(f"[BizHawk] Attempting to connect to {self.host}:{self.port} (attempt {attempt + 1}/{max_retries})")
#                 test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#                 test_socket.settimeout(2)
#                 test_socket.connect((self.host, self.port))
#                 test_socket.close()
#                 self.connected = True
#                 print(f"[BizHawk] Successfully connected to {self.host}:{self.port}")
#                 return True
#             except Exception as e:
#                 print(f"[BizHawk] Connection attempt {attempt + 1} failed: {e}")
#                 if attempt < max_retries - 1:
#                     print(f"[BizHawk] Waiting {retry_delay} seconds before retry...")
#                     time.sleep(retry_delay)
        
#         print(f"[BizHawk] Failed to connect after {max_retries} attempts")
#         return False

#     def read_memory(self, address, size):
#         # Read memory from BizHawk.
#         if not self.connected:
#             if not self.connect():
#                 raise RuntimeError("Cannot connect to BizHawk Lua bridge")
        
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#                 s.settimeout(5)  # 5 second timeout
#                 s.connect((self.host, self.port))
#                 s.sendall(f"read {address:x} {size}\n".encode())
                
#                 data = b''
#                 while len(data) < size:
#                     chunk = s.recv(size - len(data))
#                     if not chunk:
#                         break
#                     data += chunk
                
#                 if len(data) != size:
#                     raise RuntimeError(f"Expected {size} bytes, got {len(data)}")
#                 return data
#         except Exception as e:
#             print(f"[BizHawk] Memory read error: {e}")
#             self.connected = False  # Mark as disconnected on error
#             raise 