from main_server_structure import TCPServer

# Server configuration
HOST = '203.252.106.123'  # Server IP address
PORT = 23458             # Server port

# GPU SERVER IP address & Port number
GPU_SERVER_INFO = [['203.252.106.123', '203.252.106.123'],[23459, 23500]]

# Create and start the server
server = TCPServer(HOST, PORT, GPU_SERVER_INFO)
server.start()