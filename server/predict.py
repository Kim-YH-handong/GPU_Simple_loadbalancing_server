import subprocess

def predict(available_gpu, folder_addr):
    if available_gpu == 0:
        subprocess.run(['python', '/home/younghun/IoT/server/test0.py', '--volume_path', folder_addr], check = True, capture_output=True)
    elif available_gpu == 1:
        subprocess.run(['python', '/home/younghun/IoT/server/test1.py', '--volume_path', folder_addr], check = True, capture_output=True)
    elif available_gpu == 2:
        subprocess.run(['python', '/home/younghun/IoT/server/test2.py', '--volume_path', folder_addr], check = True, capture_output=True)


def send_image(image_path, name, conn):
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

