import socket
import os
import threading
import multiprocessing
from predict import predict

class TCPServer:

    gpustat = {0:True, 1:True, 2:True, 3:True, 4:True, 5:True, 6:True, 7:True}

    def __init__(self, host, port, buffer_size=4096):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.server_socket = None

    def start(self):
        # Create a TCP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")

        while True:
            # Accept a client connection
            conn, addr = self.server_socket.accept()

            # Select GPU number
            available_gpu = 'None'
            for gpu in list(TCPServer.gpustat.keys()):
                if TCPServer.gpustat[gpu] == True:
                    TCPServer.gpustat[gpu] = False
                    available_gpu = gpu
                    break
            
            # # Create a new thread to handle the client connection
            thread = threading.Thread(target=self.handle_client, args=(conn, addr, available_gpu))
            thread.start()
            # Create a new process to handle the client connection
            # process = multiprocessing.Process(target=self.handle_client, args=(conn, addr, available_gpu))
            # process.start()

    def handle_client(self, conn, addr, available_gpu):
        print(f"Connected to client at {addr[0]}:{addr[1]} with {available_gpu}")

        # Receive info
        hospital_name, patient_name, count_img = str(conn.recv(self.buffer_size).decode()).split('/')
        print(f"Patient info: {hospital_name}, {patient_name}, {count_img}")
        folder_addr = hospital_name + '/' + patient_name + '/original_img'
        os.makedirs(folder_addr, exist_ok=True)
        conn.send(folder_addr.encode())

        # Receive image from the client
        for i in range(int(count_img)):
            self.receive_image(conn, folder_addr)
        
        folder_addr = folder_addr.rsplit('original_img', 1)[0].rstrip('/')
        
        print("Check: ", folder_addr)

        predict(available_gpu, folder_addr)

        print(f"Finishing prediction: {available_gpu}")

        # Close the connection with the client
        TCPServer.gpustat[available_gpu] = True # Convert it to True

        # Return predicted images
        # TODO: return predicted images back to client

        conn.close()

    def receive_image(self, conn, folder_addr):
        # Receive the image size from the client
        size, name = str(conn.recv(self.buffer_size).decode('utf-8', errors='replace')).split('/')
        size = int(size)
        # Receive the image data from the client
        received_data = b""
        while len(received_data) < size:
            data = conn.recv(self.buffer_size)
            received_data += data

        # Save the received image
        with open(os.path.join(folder_addr, name), 'wb') as file:
            file.write(received_data)

        conn.send(name.encode())
        print("Received: ", name)

    def stop(self):
        if self.server_socket:
            self.server_socket.close()

# Server configuration
HOST = '203.252.112.20'  # Server IP address
PORT = 23458             # Server port

# Create and start the server
server = TCPServer(HOST, PORT)
server.start()
