import socket
import os
import threading
import multiprocessing
LOD = 'LOD: Request for GPU current load'


class TCPServer:

    def __init__(self, host, port, gpu_server_info, buffer_size=4096):
        # Main server
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.server_socket = None
        # Gpu0 server connection
        self.gpu0_server_host = gpu_server_info[0][0]
        self.gpu0_server_port = gpu_server_info[1][0]
        self.gpu0_server_socket = None
        # Gpu1 server connection
        self.gpu1_server_host = gpu_server_info[0][1]
        self.gpu1_server_port = gpu_server_info[1][1]
        self.gpu1_server_socket = None

    def gpu_server_start(self):
        # Connect to GPU servers
        self.gpu0_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gpu0_server_socket.connect((self.gpu0_server_host, self.gpu0_server_port))
        self.gpu1_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gpu1_server_socket.connect((self.gpu1_server_host, self.gpu1_server_port))

    def start(self):
        # Create a TCP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")

        while True:
            # Accept a client connection
            print("WAITING FOR NEW CONNECTION")
            conn, addr = self.server_socket.accept()
            self.gpu_server_start()
        
            # Create a new thread to handle the client connection
            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.start()

    def handle_client(self, conn, addr):
        #TODO: Send packets to gpu servers and Receive response.
        self.gpu0_server_socket.send(LOD.encode())
        gpu0_current_load = int(self.gpu0_server_socket.recv(self.buffer_size).decode())
        self.gpu1_server_socket.send(LOD.encode())
        gpu1_current_load = int(self.gpu1_server_socket.recv(self.buffer_size).decode())

        #TODO: Check each gpu server and find one available server.
        if gpu0_current_load <= gpu1_current_load:
            message = f'{self.gpu0_server_host}, {self.gpu0_server_port}'
            print(f"[SEND] {self.gpu0_server_host}, {self.gpu0_server_port}")
        else:
            message = f'{self.gpu1_server_host}, {self.gpu1_server_port}'
            print(f"[SEND] {self.gpu1_server_host}, {self.gpu1_server_port}")
        
        #TODO: Send available gpu server's ip address & port number.
        conn.send(message.encode())
        # conn.close()

    def stop(self):
        if self.server_socket:
            self.server_socket.close()
