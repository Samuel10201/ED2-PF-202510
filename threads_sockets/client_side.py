import socket
import json

SERVER = "10.20.25.51"  # IP del servidor
PORT = 8080

# Leer el JSON desde archivo
with open('resultado2.json', 'r', encoding='utf-8') as f:
    json_data = f.read()

json_bytes = json_data.encode('utf-8')
json_length = len(json_bytes)

# Conecta al servidor
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((SERVER, PORT))

client.sendall(f"{json_length:010}".encode('utf-8'))  # Por ejemplo: b'0000012345'
client.sendall(json_bytes)

# Espera la respuesta
response = client.recv(1024).decode('utf-8')
print("From Server:", response)

client.sendall(b'bye')
client.close()

