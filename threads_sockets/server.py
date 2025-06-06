"""
server.py

Este módulo actúa como un servidor TCP que:
1. Se conecta a una base de datos MySQL.
2. Extrae un conjunto de datos y lo convierte en un DataFrame.
3. Escucha conexiones entrantes de múltiples clientes simultáneamente.
4. Envía el DataFrame serializado a cada cliente.
5. Recibe resultados procesados por los clientes.

"""

# 📦 Importación de librerías necesarias para la conexión, serialización, concurrencia, visualización y manejo de datos
import socket
import threading
import pickle
import pandas as pd
import time
import os
import json
import pyarrow as pa
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

# 📁 Carga de variables de entorno (credenciales y configuración de la base de datos)
load_dotenv()

# 📌 Configuración para la conexión a la base de datos
# Se llaman las variables de entorno para evitar exponer credenciales sensibles
config = {
    "user": environ['DATABASE_USERNAME'],
    "password": environ['DATABASE_PASSWORD'],
    "host": environ['DATABASE_HOST'],
    "database": environ['DATABASE_NAME'],
    "charset": 'utf8'
}

# 🔌 Función para establecer la conexión a la base de datos MySQL
def get_connection():
    """
    Establece una conexión con la base de datos MySQL utilizando
    las credenciales cargadas desde las variables de entorno.

    Returns:
        mysql.connector.connection.MySQLConnection: Objeto de conexión activa,
        o None si ocurre un error.
    """
    try:
        print("Connecting to the database...\n config: ", config)
        return connect(**config)
    except Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("❌ Usuario o contraseña incorrectos")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("❌ La base de datos no existe")
        else:
            print("❌ Error desconocido:", err)
        return None

# 🧮 Función que ejecuta una consulta y devuelve los resultados
def get_data(connection: connect, query: str):
    """
    Ejecuta una consulta SQL en la base de datos proporcionada y
    devuelve los resultados.

    Args:
        connection (mysql.connector.connection.MySQLConnection): Conexión activa a la base de datos.
        query (str): Consulta SQL a ejecutar.

    Returns:
        list[tuple]: Resultados obtenidos de la consulta.
    """
    my_cursor = connection.cursor()
    my_cursor.execute(query)
    data = my_cursor.fetchall()
    my_cursor.close()
    return data

# 📋 Función que carga todos los datos desde la base de datos y los convierte en un DataFrame
def get_dataframe():
    """
    Carga los datos desde la base de datos MySQL y los convierte en un DataFrame de Pandas.

    Returns:
        pd.DataFrame: DataFrame que contiene los registros de la tabla UN.VENTAS.
    """
    cnx = get_connection()
    print("[📥] Cargando datos desde la base de datos...")

    inicio = time.time()
    data = get_data(cnx, "SELECT * FROM UN.VENTAS")
    print(time.time() - inicio)  # Tiempo que tarda la consulta

    # Definir nombres de columnas esperados para el DataFrame
    df = pd.DataFrame(data, columns=[
        'ID_VENTA', 'FECHA_VENTA', 'ID_CLIENTE', 'ID_EMPLEADO',
        'ID_PRODUCTO', 'CANTIDAD', 'PRECIO_UNITARIO', 'DESCUENTO', 'FORMA_PAGO'
    ])
    cnx.close()
    print(f"[✅] Datos cargados: {len(df)} registros.")
    return df
def graficar_resultados(formats_dict, client_id):
    """
    Genera una gráfica de barras que muestra los tiempos de serialización para
    distintos formatos.

    Args:
        formats_dict (dict): Diccionario con nombre del formato como clave y tiempo en segundos como valor.
        client_id (str): Nombre o identificador del cliente que se está graficando.
    """
    formatos = list(formats_dict.keys())
    tiempos = list(formats_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(formatos, tiempos, color='skyblue')
    plt.ylabel('Tiempo de serialización (s)')
    plt.title(f'Tiempos de serialización por formato - Cliente {client_id}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = f"grafica_serializacion_{client_id}.png"
    plt.savefig(filename)
    print(f"[📊] Gráfica guardada como: {filename}")
    plt.close()

# 🤝 Función que maneja la conexión con un cliente específico
def handle_client(conn, addr, df):
    """
    Maneja la conexión con un cliente:
    - Envía un DataFrame serializado.
    - Espera y recibe resultados procesados desde el cliente.

    Args:
        conn (socket.socket): Objeto de conexión con el cliente.
        addr (tuple): Dirección IP y puerto del cliente.
        df (pd.DataFrame): DataFrame que será enviado al cliente.
    """
    print(f"[+] Conectado con {addr}")

    # 🔄 Serializar el DataFrame
    data_bytes = pickle.dumps(df)

    # Enviar primero el tamaño del mensaje para que el cliente sepa cuánto esperar
    conn.sendall(len(data_bytes).to_bytes(8, 'big'))
    conn.sendall(data_bytes)

    # ⬅️ Esperar tamaño del mensaje de respuesta
    length = int.from_bytes(conn.recv(8), 'big')

    # Recibir y deserializar los resultados
    results = pickle.loads(conn.recv(length))
    
        # 📈 Mostrar resultados procesados recibidos del cliente
    print(f"[{addr}] Resultados recibidos:\n", results)

    # 🔍 Si contiene tiempos de serialización, graficarlos
    if isinstance(results, dict) and "format_times" in results and "client_id" in results:
        graficar_resultados(results["format_times"], results["client_id"])


    # 📈 Mostrar resultados procesados recibidos del cliente
    print(f"[{addr}] Resultados recibidos:\n", results)

    conn.close()

# 🚀 Función principal que inicia el servidor y acepta múltiples conexiones
def start_server(df):
    """
    Inicia el servidor, escucha en un puerto definido y acepta múltiples
    conexiones de clientes, delegando cada una a un hilo nuevo.

    Args:
        df (pd.DataFrame): DataFrame que será enviado a cada cliente conectado.
    """
    HOST = '0.0.0.0'  # Escucha en todas las interfaces de red disponibles
    PORT = 9999       # Puerto en el que se escucha

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("[🟢] Servidor escuchando en el puerto", PORT)

        while True:
            conn, addr = s.accept()
            # 🧵 Crear un nuevo hilo para manejar la conexión de este cliente
            thread = threading.Thread(target=handle_client, args=(conn, addr, df))
            thread.start()
            print(f"[👥] Número de conexiones activas: {threading.active_count() - 1}")

# 🏁 Punto de entrada principal del servidor
if __name__ == "__main__":
    df = get_dataframe()   # Obtener datos desde la base de datos
    start_server(df)       # Iniciar servidor con el DataFrame cargado