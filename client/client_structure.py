import socket
import os

class TCPClient_Main:
    def __init__(self, server_host, server_port, buffer_size=4096):
        self.server_host = server_host
        self.server_port = server_port
        self.buffer_size = buffer_size
        self.client_socket = None
        
    def connect(self):
        # Create a TCP socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_host, self.server_port))
        print(f"Connected to server at {self.server_host}:{self.server_port}")

    def get_gpu_server(self):
        # Packet from Main Server will be look like {203.252.112.20, 23456} 
        return str(self.client_socket.recv(self.buffer_size).decode()).split(',')

    def close(self):
        if self.client_socket:
            self.client_socket.close()