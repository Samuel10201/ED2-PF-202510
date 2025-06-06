import socket
import threading
import json

class ClientThread(threading.Thread):
    def __init__(self, clientAddress, clientsocket):
        threading.Thread.__init__(self)
        self.csocket = clientsocket
        self.clientAddress = clientAddress
        print("New connection added:", clientAddress)

    def run(self):
        print("Connection from:", self.clientAddress)
        try:
            # Recibe primero los 10 bytes con el tamaño
            size_data = self.csocket.recv(10)
            if not size_data:
                print("No size data received")
                return

            total_size = int(size_data.decode('utf-8'))

            # Ahora recibe exactamente `total_size` bytes
            buffer = b""
            while len(buffer) < total_size:
                chunk = self.csocket.recv(min(4096, total_size - len(buffer)))
                if not chunk:
                    break
                buffer += chunk

            # Intenta cargar el JSON
            try:
                json_str = buffer.decode('utf-8')
                data_json = json.loads(json_str)
                print(f"Received JSON from client {self.clientAddress}:")
                print(json.dumps(data_json, indent=2))

                # Guardar el JSON en un archivo
                with open('ventas_recibidas.json', 'w', encoding='utf-8') as f:
                    json.dump(data_json, f, indent=4, ensure_ascii=False)
                print("JSON guardado como ventas_recibidas.json")

                response = "JSON received OK"
            except json.JSONDecodeError as e:
                print("Error decodificando JSON:", e)
                response = "Invalid JSON received"

            self.csocket.send(response.encode('utf-8'))

        finally:
            print("Client at", self.clientAddress, "disconnected")
            self.csocket.close()

LOCALHOST = "10.20.25.51"
PORT = 8080

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((LOCALHOST, PORT))
server.listen(5)

print("Server started, waiting for connections...")

while True:
    clientsock, clientAddress = server.accept()
    newthread = ClientThread(clientAddress, clientsock)
    newthread.start()