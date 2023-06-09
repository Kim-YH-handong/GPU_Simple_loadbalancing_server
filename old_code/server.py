import socket
import os

# Server configuration
HOST = '203.252.106.123' # Server IP address
PORT = 23457       # Server port
BUFFER_SIZE = 4096  # Size of the buffer for receiving data

# Function to receive an image from the client
def receive_image(conn, folder_addr):
    # Receive the image size from the client
    size, name = str(conn.recv(BUFFER_SIZE).decode()).split('/')
    size = int(size)

    # Receive the image data from the client
    received_data = b""
    while len(received_data) < size:
        data = conn.recv(BUFFER_SIZE)
        received_data += data

    # Save the received image
    with open(os.path.join(folder_addr, name), 'wb') as file:
        file.write(received_data)
    
    conn.send(name.encode())
    print("Received: ", name)

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the specified address and port
server_socket.bind((HOST, PORT))

# Listen for incoming connections
server_socket.listen(1)
print(f"Server listening on {HOST}:{PORT}")

while True:
    # Accept a client connection
    conn, addr = server_socket.accept()
    print(f"Connected to client at {addr[0]}:{addr[1]}")

    # Receive info
    hospital_name, patient_name, count_img = str(conn.recv(BUFFER_SIZE).decode()).split('/')
    print(f"Patient info: {hospital_name}, {patient_name}, {count_img}")
    folder_addr = hospital_name + '/' + patient_name + '/original_img' 
    os.makedirs(folder_addr, exist_ok = True)


    # Receive image from the client
    for i in range(int(count_img)):
        receive_image(conn, folder_addr)

    # Close the connection with the client
    conn.close()
