from gpu_server_structure import TCPServer

# Server configuration
HOST = '203.252.106.123'   # Server IP address
PORT = 23500               # Server port

# Create and start the server
server = TCPServer(HOST, PORT)
server.start()
