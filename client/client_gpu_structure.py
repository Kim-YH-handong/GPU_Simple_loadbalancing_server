import socket
import os
ACK = 'ACK: Images Sent Finish'
USE = 'USE: Request for GPU Use'
IMAGE_INFO_ACK = "ACK: Image information arrived"


class TCPClient:
    def __init__(self, server_host, server_port, hospital_name, patient_name, buffer_size=4096):
        self.server_host = server_host
        self.server_port = server_port
        self.buffer_size = buffer_size
        self.client_socket = None
        self.hospital_name = hospital_name
        self.patient_name = patient_name
        self.files = sorted(os.listdir(patient_name))

    def connect(self):
        # Create a TCP socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_host, int(self.server_port)))
        print(f"Connected to server at {self.server_host}:{self.server_port}")

    def start(self):
        #TODO: Send that it is sending images
        self.client_socket.send(USE.encode())

        #TODO: Check USE allow
        ack_message = self.client_socket.recv(self.buffer_size).decode()
        print(f'[RECEIVE] ACK for USE Packet')

        #TODO: Send folder info
        self.client_socket.send(str(os.path.join(self.hospital_name, self.patient_name)).encode())
        print(f"[SEND] {self.hospital_name} / {self.patient_name}")
        ack_message = self.client_socket.recv(self.buffer_size).decode()
        print(f"[RECEIVE] {ack_message}")        

        #TODO: Send images
        for name in self.files:
            self.send_image(os.path.join(self.patient_name, name), name)
        self.client_socket.send(ACK.encode())

        #TODO: Wait for GPU USE ACK from gpu server
        print("WAITING for GPU SERVER RESPONSE")
        # PohangS/person3/result_img
        folder_addr = self.client_socket.recv(self.buffer_size).decode()
        os.makedirs(folder_addr, exist_ok=True)
        print(f"[ACK] {folder_addr}")

        #TODO: Receive images until ACK come
        while True:
            received_message = str(self.client_socket.recv(self.buffer_size).decode())
            if received_message[:3] == 'ACK':
                print(f"[RECEIVED] {received_message}")
                break
            else:
                self.receive_image(folder_addr, received_message)
        
    def send_image(self, image_path, name):
        # Read the image file
        with open(image_path, 'rb') as file:
            image_data = file.read()

        # Send the image size and name to the server PACKET_EX) 89893/CLAHE_김영헌_semisolid_00001.jpeg
        self.client_socket.send(str(os.path.join(str(len(image_data)), name)).encode()) # Send Image info
        print(f"[SEND] INFO: {len(image_data)}, {name}")
        self.client_socket.recv(self.buffer_size).decode() # Wait for ACK 
        print(f"[RECEIVE] ACK for INFO")
        self.client_socket.sendall(image_data) # Send Image
        print(f"[SEND] PATH: {image_path}")
        finish = str(self.client_socket.recv(self.buffer_size).decode())
        print(f"[RECEIVE] ACK: {finish}")

    def receive_image(self, folder_addr, received_message):
        # Receive the image size from the client
        size, name = received_message.split('/')
        self.client_socket.send(IMAGE_INFO_ACK.encode())

        # Receive the image data from the client
        received_data = b""
        while len(received_data) < int(size):
            data = self.client_socket.recv(self.buffer_size)
            received_data += data

        # Receive succssfully
        self.client_socket.send(name.encode())

        # Save the received image
        with open(os.path.join(folder_addr, name), 'wb') as file:
            file.write(received_data)

        print(f"[RECEIVED] {name}")

    def send_patient_info(self, hospital_name, patient_name, num_images):
        # Send hospitalname/patientname/#ofimages to the server
        # PACKET_EX) SeoulNationalHospital/김영헌/15
        packet_info = hospital_name + '/' + patient_name + '/' + str(num_images)
        self.client_socket.send(packet_info.encode('utf-8'))
        finish = str(self.client_socket.recv(self.buffer_size).decode())
        print(f"[SUCCESS]: {finish}")

    def close(self):
        if self.client_socket:
            self.client_socket.close()