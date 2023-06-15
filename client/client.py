import os
from client_structure import TCPClient_Main
from client_gpu_structure import TCPClient

# Client configuration
SERVER_HOST = '203.252.106.123'  # Server IP address
SERVER_PORT = 23458           # Server port
BUFFER_SIZE = 4096              # Size of the buffer for receiving data
HOSPITAL_NAME = 'PohangS'
PATIENT_NAME = input("Enter address of patient: ")

# Create and connect the client
client = TCPClient_Main(SERVER_HOST, SERVER_PORT, BUFFER_SIZE)
client.connect()
GPU_SERVER_HOST, GPU_SERVER_PORT = client.get_gpu_server()
print(f"[Response] GPU SERVER ALLOCATION: {GPU_SERVER_HOST}, {GPU_SERVER_PORT}")
client.close()

client = TCPClient(GPU_SERVER_HOST, GPU_SERVER_PORT, HOSPITAL_NAME, PATIENT_NAME,  BUFFER_SIZE)
client.connect()
client.start()
print("FINISH!!!")
client.close()

# Close the connection with the server
client.close()
