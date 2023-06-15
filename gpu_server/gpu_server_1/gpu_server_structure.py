import socket
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import threading
import multiprocessing
import subprocess
import torch
import torch.nn as nn
import nvidia_smi

INDEX = 1
IMAGE_INFO_ACK = "ACK: Image information arrived"
PREDICTION_ACK = "ACK: Prediction End"
USE_ACK = "ACK: Use GPU Ready!"
ACK = 'ACK: Images Sent Finish'

class TCPServer:

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
            print("WAITING FOR NEW CONNECTION!")
            # Accept a client connection
            conn, addr = self.server_socket.accept()
            print("[Request] Request arrived")
            request_type = str(conn.recv(self.buffer_size).decode().split(':')[0])
            
            if request_type == 'LOD':
                #TODO: Get current GPU status and send
                print("[Request] Get Current Load")
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(INDEX)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                send_message = str(info.used)
                conn.send(send_message.encode()) 
                print(f"[Response:0] Send current load {send_message}")
            elif request_type == 'USE':
                #TODO: Call handle_client
                print("[Request] Use GPU")
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.start()
            else:
                print("[Request] Wrong request format")

    def handle_client(self, conn, addr):
        print(f"[GPU CONNECTION ESTABLISHMENT] {addr[0]}:{addr[1]}")
        #TODO: Send ACK for USE
        conn.send(USE_ACK.encode())
        print(f"[SEND] {USE_ACK}")

        #TODO: Receive folder address and create the folder and send ACK
        hospital_name, patient_name = str(conn.recv(self.buffer_size).decode()).split('/')
        print(f"[RECEIVED] {hospital_name}, {patient_name}")
        conn.send(IMAGE_INFO_ACK.encode())
        print(f"[SEND] {IMAGE_INFO_ACK}")

        #TODO: Receive images until ACK come
        folder_addr = hospital_name + '/' + patient_name + '/original_img'
        os.makedirs(folder_addr, exist_ok=True)
        while True:
            received_message = str(conn.recv(self.buffer_size).decode())
            if received_message[:3] == 'ACK':
                print(f"[RECEIVED] {received_message}")
                break
            else:
                self.receive_image(conn, folder_addr, received_message)
        
        #TODO: Run AI model
        folder_addr = os.path.join(hospital_name, patient_name)
        print(f"[RUN GPU START] Test code start! folder_addr = {folder_addr}")
        subprocess.run(['python', '/home/younghun/IoT/gpu_server/gpu_server_0/test0.py', '--volume_path', folder_addr], check = True, capture_output=True)
        print(f"[RUN GPU END] Test code END!")

        #TODO: SEND ACK
        folder_addr = hospital_name + '/' + patient_name + '/result_img' 
        conn.send(folder_addr.encode())

        #TODO: SEND predicted images
        for name in os.listdir(folder_addr):
            self.send_image(conn, os.path.join(folder_addr, name), name)
        conn.send(ACK.encode())

    def send_image(self, conn, image_path, name):
        # Read the image file
        with open(image_path, 'rb') as file:
            image_data = file.read()

        # Send the image size and name to the server PACKET_EX) 89893/CLAHE_김영헌_semisolid_00001.jpeg
        conn.send(str(os.path.join(str(len(image_data)), name)).encode()) # Send Image info
        print(f"[SEND] INFO: {len(image_data)}, {name}")
        conn.recv(self.buffer_size).decode() # Wait for ACK 
        print(f"[RECEIVE] ACK for INFO")
        conn.sendall(image_data) # Send Image
        print(f"[SEND] PATH: {image_path}")
        finish = str(conn.recv(self.buffer_size).decode())
        print(f"[RECEIVE] ACK: {finish}")

    def receive_image(self, conn, folder_addr, received_message):
        # Receive the image size from the client
        size, name = received_message.split('/')
        conn.send(IMAGE_INFO_ACK.encode())

        # Receive the image data from the client
        received_data = b""
        while len(received_data) < int(size):
            data = conn.recv(self.buffer_size)
            received_data += data

        # Receive succssfully
        conn.send(name.encode())

        # Save the received image
        with open(os.path.join(folder_addr, name), 'wb') as file:
            file.write(received_data)

        print(f"[RECEIVED] {name}")

    def stop(self):
        if self.server_socket:
            self.server_socket.close()