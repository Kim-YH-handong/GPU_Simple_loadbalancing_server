import socket
import os

class TCPClient:
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

    def send_image(self, image_path, name):
        # Read the image file
        with open(image_path, 'rb') as file:
            image_data = file.read()

        # Send the image size and name to the server PACKET_EX) 89893/CLAHE_김영헌_semisolid_00001.jpeg
        send_info = str(len(image_data)) + '/' + name
        send_info_utf8 = send_info.encode('utf-8', errors='replace')
        self.client_socket.send(send_info_utf8)
        # self.client_socket.send(send_info.encode('utf-8'))

        # Send the image data to the server
        self.client_socket.sendall(image_data)
        print("Sent: ", image_path)

    def send_patient_info(self, hospital_name, patient_name, num_images):
        # Send hospitalname/patientname/#ofimages to the server
        # PACKET_EX) SeoulNationalHospital/김영헌/15
        packet_info = hospital_name + '/' + patient_name + '/' + str(num_images)
        self.client_socket.send(packet_info.encode('utf-8'))
        finish = str(self.client_socket.recv(self.buffer_size).decode())
        print("INFO_Success: ", finish)

    def close(self):
        if self.client_socket:
            self.client_socket.close()

# Client configuration
SERVER_HOST = '203.252.112.20'  # Server IP address
SERVER_PORT = 23458           # Server port
BUFFER_SIZE = 4096              # Size of the buffer for receiving data
HOSPITAL_NAME = 'PohangS'
PATIENT_NAME = input("Enter address of patient: ")

# Create and connect the client
client = TCPClient(SERVER_HOST, SERVER_PORT, BUFFER_SIZE)
client.connect()

# Send patient info to the server
files = sorted(os.listdir(PATIENT_NAME))
num_images = len(files)
client.send_patient_info(HOSPITAL_NAME, PATIENT_NAME, num_images)

# Send images to the server
for name in files:
    image_path = os.path.join(PATIENT_NAME, name)
    client.send_image(image_path, name)
    finish = str(client.client_socket.recv(BUFFER_SIZE).decode())
    print("Success: ", finish)

print("Waiting for server's response")

# Close the connection with the server
client.close()
