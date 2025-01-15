import socket
import pickle
import numpy as np
import cv2
from multiprocessing import Pool

def gaussian_kernel(size, sigma=1):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for i in range(size):
        for j in range(size):
            diff = (i - center) ** 2 + (j - center) ** 2
            kernel[i, j] = np.exp(-diff / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel

def apply_gaussian_blur(image, kernel):
    img_height, img_width = image.shape[:2]
    k_size = kernel.shape[0]
    pad = k_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    blurred_image = np.zeros_like(image)
    
    for i in range(img_height):
        for j in range(img_width):
            for c in range(3):
                blurred_image[i, j, c] = np.sum(
                    kernel * padded_image[i:i + k_size, j:j + k_size, c]
                )

    return blurred_image

def send_all(sock, data):
    data = pickle.dumps(data)
    sock.sendall(len(data).to_bytes(4, byteorder='big'))
    sock.sendall(data)

def receive_all(sock):
    try:
        length = int.from_bytes(sock.recv(4), byteorder='big')
        if not length:
            return None
        data = b''
        while len(data) < length:
            packet = sock.recv(4096)
            if not packet:
                return None
            data += packet
        return pickle.loads(data)
    except EOFError:
        print("Error: Received incomplete data.")
        return None

def blur (img):
    blurred_img = apply_gaussian_blur(img, kernel)
    return blurred_img

kernel = gaussian_kernel(15, sigma=3)
parts_count = 4

def start_server(host, port):

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print("Server is listening...")
        
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")

        with conn:
            while True:
                fragment = receive_all(conn)
                if fragment is None:
                    print("No data received or connection closed.")
                    break
                
                print("Fragment received from client.")
                
                height, width, channels = fragment.shape
                part_width = int(width / parts_count)
                parts = []

                print("Blurring with CPU.")

                for i in range(0, parts_count):
                    parts.append(fragment[0:height, (0 + i * part_width):(part_width + i * part_width)])


                with Pool(4) as p:
                    results = p.map(blur, parts)
                
                blurred_fragment = np.hstack(results)
                print("Fragment blurred.")
                
                send_all(conn, blurred_fragment)
                print("Blurred fragment sent back to client.")

if __name__ == "__main__":
    while 1:
        start_server('0.0.0.0', 65432)  # Ustaw adres IP serwera
