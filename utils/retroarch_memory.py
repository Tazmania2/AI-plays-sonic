# DEPRECATED: This file is no longer used. See utils/bizhawk_memory_file.py for BizHawk integration.

# import socket
# import struct
# import time

# class RetroArchMemoryReader:
#     def __init__(self, host='127.0.0.1', port=55355):
#         self.host = host
#         self.port = port
#         self.sock = None
#         self.connected = False

#     def connect(self, max_retries=10, retry_delay=1):
#         # Connect to RetroArch with retry logic.
#         # Try different host addresses
#         hosts_to_try = ['127.0.0.1', 'localhost', '0.0.0.0']
        
#         for host in hosts_to_try:
#             print(f"[MemoryRead] Trying host: {host}")
#             for attempt in range(max_retries):
#                 try:
#                     if self.sock:
#                         self.sock.close()
                    
#                     self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#                     self.sock.settimeout(5)  # 5 second timeout
#                     print(f"[MemoryRead] Attempting to connect to RetroArch on {host}:{self.port} (attempt {attempt+1}/{max_retries})")
#                     self.sock.connect((host, self.port))
#                     self.connected = True
#                     self.host = host
#                     print(f"[MemoryRead] Successfully connected to RetroArch on {host}:{self.port}!")
#                     return True
#                 except Exception as e:
#                     print(f"[MemoryRead] Connection attempt {attempt+1} failed on {host}:{self.port}: {e}")
#                     if attempt < max_retries - 1:
#                         print(f"[MemoryRead] Waiting {retry_delay} seconds before retry...")
#                         time.sleep(retry_delay)
#                     else:
#                         print(f"[MemoryRead] Failed to connect to {host}:{self.port} after {max_retries} attempts")
        
#         print(f"[MemoryRead] Failed to connect to any host after trying all addresses")
#         self.connected = False
#         return False

#     def ensure_connected(self):
#         # Ensure we have a connection, reconnect if needed.
#         if not self.connected or not self.sock:
#             return self.connect()
#         return True

#     def read_memory(self, address, size):
#         # Read memory from RetroArch.
#         if not self.ensure_connected():
#             raise RuntimeError("Could not connect to RetroArch")
        
#         try:
#             # Build the command: READ_CORE_MEMORY <address> <size>\n
#             cmd = f'READ_CORE_MEMORY {address:X} {size}\n'.encode('ascii')
#             self.sock.sendall(cmd)
            
#             # RetroArch returns the data as raw bytes
#             data = b''
#             while len(data) < size:
#                 chunk = self.sock.recv(size - len(data))
#                 if not chunk:
#                     break
#                 data += chunk
            
#             if len(data) != size:
#                 raise RuntimeError(f"Expected {size} bytes, got {len(data)}")
            
#             return data
#         except Exception as e:
#             print(f"[MemoryRead] Error reading memory at {address:X}: {e}")
#             # Try to reconnect on error
#             self.connected = False
#             raise

#     def close(self):
#         # Close the connection.
#         if self.sock:
#             self.sock.close()
#             self.sock = None
#             self.connected = False 