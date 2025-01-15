import socket
import pickle
import numpy as np
import cv2
from threading import Thread
import math
import io
import requests

def send_all(sock, data):
    data = pickle.dumps(data)
    sock.sendall(len(data).to_bytes(4, byteorder='big'))
    sock.sendall(data)

def receive_all(sock):
    length = int.from_bytes(sock.recv(4), byteorder='big')
    data = b''
    while len(data) < length:
        packet = sock.recv(4096)
        if not packet:
            break
        data += packet
    return pickle.loads(data)

def send_fragment_to_server(url, fragment):
    """Sends a fragment to the server and returns the processed fragment."""
    # Encode the fragment as an image (e.g., PNG format) instead of pickling
    _, buffer = cv2.imencode('.png', fragment)
    data = io.BytesIO(buffer.tobytes())
    
    # Send the fragment via HTTP POST request as a file
    files = {'image': ('fragment.png', data, 'image/png')}
    response = requests.post(url, files=files)
    
    # Deserialize the processed fragment
    if response.status_code == 200:
        processed_fragment = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        return processed_fragment
    else:
        print(f"Failed to process fragment: {response.status_code}")
        return None

def process_fragment_on_server(server_ip, server_port, fragments, results, server_index):
    if server_port == -1:
        """Sends fragments to the server and receives processed fragments."""
        for i, fragment in enumerate(fragments):
            print(f"Sending fragment {i + server_index} to server at {server_ip}")
            processed_fragment = send_fragment_to_server(server_ip, fragment)
            
            if processed_fragment is not None:
                print(f"Processed fragment {i + server_index} received from server at {server_ip}")
                results[i + server_index] = processed_fragment
            else:
                print(f"Error processing fragment {i + server_index}")
    else:
        """Wysyła fragmenty do serwera i odbiera przetworzone fragmenty."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((server_ip, server_port))
            print(f"Connected to server {server_ip}:{server_port}")
            
            for i, fragment in enumerate(fragments):
                send_all(client_socket, fragment)
                print(f"Fragment {i + server_index} sent to server {server_ip}")
                
                processed_fragment = receive_all(client_socket)
                print(f"Processed fragment {i + server_index} received from server {server_ip}")
                
                # Zapisz przetworzony fragment w wynikach, by zachować kolejność
                results[i + server_index] = processed_fragment

def start_client(server_info, image_path, parts_count=16):
    # Wczytanie i podzielenie obrazu na fragmenty
    img = cv2.imread(image_path)
    #img = cv2.resize(img, (800, 600))
    height, width, channels = img.shape
    part_width = width // parts_count
    parts = [img[0:height, i * part_width:(i + 1) * part_width] for i in range(parts_count)]

    # Przygotowanie struktur dla wyników
    results = [None] * parts_count

    # Obliczanie liczby fragmentów na serwer
    fragments_per_server = math.ceil(parts_count / len(server_info))
    
    server_threads = []
    for idx, (server_ip, server_port) in enumerate(server_info):
        # Ustal zakres fragmentów dla danego serwera
        start_idx = idx * fragments_per_server
        end_idx = min(start_idx + fragments_per_server, parts_count)
        server_fragments = parts[start_idx:end_idx]

        # Start nowego wątku dla każdego serwera
        thread = Thread(target=process_fragment_on_server, args=(server_ip, server_port, server_fragments, results, start_idx))
        server_threads.append(thread)
        thread.start()

    # Czekaj na zakończenie wszystkich wątków
    for thread in server_threads:
        thread.join()

    # Upewnij się, że wszystkie fragmenty zostały przetworzone
    if any(fragment is None for fragment in results):
        print("Error: Not all fragments were processed successfully.")
        return

    # Scalanie przetworzonych fragmentów
    final_image = np.hstack(results)
    cv2.imwrite('blurred_image.png', final_image)
    print("Final blurred image saved as 'blurred_image.png'.")

# Przykładowe użycie
if __name__ == "__main__":
    # Lista serwerów w formie (IP, PORT) lub (http://IP:PORT/api/image/blur, -1)
    server_info = [
        ('xxx.xxx.xxx.xxx', 65432),
        ('http://xxx.xxx.xxx.xxx:8080/api/image/blur', -1)
    ]
    start_client(server_info, './tap.png', parts_count=4)

