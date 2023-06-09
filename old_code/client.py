import socket
import os

# Server configuration
SERVER_HOST = '203.252.106.123'  # Server IP address
SERVER_PORT = 23457       # Server port
BUFFER_SIZE = 4096        # Size of the buffer for receiving data
HOSPITAL_NAME = 'PohangS'
PATIENT_NAME = input("Enter address of patient: ")

# Function to send an image to the server
def send_image(conn, image_path, name):
    # Read the image file
    with open(image_path, 'rb') as file:
        image_data = file.read()

    # Send the image size to the server PACKET_EX) 89893/CLAHE_김영헌_semisolid_00001.jpeg
    send_info = str(len(image_data)) + '/' + name
    conn.send(send_info.encode())

    # Send the image data to the server
    conn.sendall(image_data)
    print("Sent: ", image_path)

# Create a TCP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((SERVER_HOST, SERVER_PORT))
print(f"Connected to server at {SERVER_HOST}:{SERVER_PORT}")

# Send hospitalname/patientname/#ofimages to the server
# PACKET_EX) SeoulNationalHospital/김영헌/15
files = os.listdir(PATIENT_NAME)
packet_info = HOSPITAL_NAME + '/' + PATIENT_NAME + '/' + str(len(files))
client_socket.send(packet_info.encode())
finish = str(client_socket.recv(BUFFER_SIZE).decode())
print("INFO_Success: ", finish)

# Send an image to the server
for name in files:
    image_path = name  # Replace with your image file path
    send_image(client_socket, os.path.join(PATIENT_NAME, image_path), name)
    finish = str(client_socket.recv(BUFFER_SIZE).decode())
    print("Success: ", finish)

print("Waiting for server's response")

# Close the connection with the server
client_socket.close()
