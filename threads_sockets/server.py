import socket
import threading
import pickle

import pandas as pd
import time
import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
from fastavro import writer, parse_schema
import matplotlib.pyplot as plt
import datetime
from decimal import Decimal
import numpy as np
from multiprocessing import Process, Manager
from mysql.connector import connect, Error, errorcode
from os import environ
from dotenv import load_dotenv
import math
import io

# Cargar configuración
load_dotenv()
config = {
    "user": environ['DATABASE_USERNAME'],
    "password": environ['DATABASE_PASSWORD'],
    "host": environ['DATABASE_HOST'],
    "database": environ['DATABASE_NAME'],
    "charset": 'utf8'
}

def get_connection():
    try:
        print("Connecting to the database...\n config: ", config)
        return connect(**config)
    except Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        return None

def get_data(connection: connect, query: str):
    my_cursor = connection.cursor()
    my_cursor.execute(query)
    data = my_cursor.fetchall()
    my_cursor.close()
    return data

def get_dataframe():
    cnx = get_connection()
    print("[📥] Cargando datos desde la base de datos...")
    inicio = time.time()
    data = get_data(cnx, "SELECT * FROM UN.VENTAS")
    print(inicio - time.time())
    
    df = pd.DataFrame(data, columns=[
        'ID_VENTA', 'FECHA_VENTA', 'ID_CLIENTE', 'ID_EMPLEADO',
        'ID_PRODUCTO', 'CANTIDAD', 'PRECIO_UNITARIO', 'DESCUENTO', 'FORMA_PAGO'
    ])
    cnx.close()
    print(f"[✅] Datos cargados: {len(df)} registros.")
    return df

def handle_client(conn, addr, df):
    print(f"[+] Conectado con {addr}")
    # Serializar y enviar el DataFrame
    data_bytes = pickle.dumps(df)
    conn.sendall(len(data_bytes).to_bytes(8, 'big'))  # tamaño primero
    conn.sendall(data_bytes)

    # Recibir resultados
    length = int.from_bytes(conn.recv(8), 'big')
    results = pickle.loads(conn.recv(length))

    print(f"[{addr}] Resultados recibidos:\n", results)
    conn.close()

def start_server(df):
    HOST = '0.0.0.0'
    PORT = 9999

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("[🟢] Servidor escuchando en el puerto", PORT)

        while True:
            conn, addr = s.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr, df))
            thread.start()
            print(f"[👥] Número de conexiones activas: {threading.active_count() - 1}")

if __name__ == "__main__":
    df = get_dataframe()
    start_server(df)
