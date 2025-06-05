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

# Cargar variables de entorno
load_dotenv()

config = {
    "user": environ['DATABASE_USERNAME'],
    "password": environ['DATABASE_PASSWORD'],
    "host": environ['DATABASE_HOST'],
    "database": environ['DATABASE_NAME'],
    "charset": 'utf8'
}

formats = ['json', 'csv', 'parquet', 'avro']
os.makedirs("output_files", exist_ok=True)

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

def serialize_data(df, fmt):
    if fmt == 'json':
        df = df.copy()
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (datetime.date, datetime.datetime))).any():
                df[col] = df[col].astype(str)
        with io.StringIO() as buffer:
            df.to_json(buffer, orient='records', lines=True)
            _ = buffer.getvalue()  # forzar la escritura

    elif fmt == 'csv':
        with io.StringIO() as buffer:
            df.to_csv(buffer, index=False)
            _ = buffer.getvalue()

    elif fmt == 'parquet':
        table = pa.Table.from_pandas(df)
        with pa.BufferOutputStream() as buf:
            pq.write_table(table, buf)

    elif fmt == 'avro':
        # Este sí necesita archivo físico, lamentablemente
        df = df.copy()
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (datetime.date, datetime.datetime))).any():
                df[col] = df[col].astype(str)
            elif df[col].apply(lambda x: isinstance(x, Decimal)).any():
                df[col] = df[col].astype(float)

        records = df.to_dict(orient='records')
        avro_types = {
            'int64': 'long',
            'float64': 'double',
            'object': 'string',
            'bool': 'boolean'
        }

        schema = {
            "type": "record",
            "name": "Venta",
            "fields": []
        }

        for col in df.columns:
            dtype = str(df[col].dtype)
            avro_type = avro_types.get(dtype, 'string')
            schema["fields"].append({
                "name": col,
                "type": ["null", avro_type],
                "default": None
            })

        # Usamos un archivo temporal que se elimina inmediatamente
        import tempfile
        with tempfile.NamedTemporaryFile(delete=True, suffix=".avro") as temp_file:
            writer(temp_file, parse_schema(schema), records)

    else:
        raise ValueError("Formato no soportado")

def medir_tiempo_promedio(fmt, df_sample, size, results):
    tiempos = []
    for i in range(10):
        start = time.time()
        serialize_data(df_sample.copy(), fmt)  # Ahora no se guarda en archivo
        tiempos.append(time.time() - start)

    promedio = sum(tiempos) / len(tiempos)
    results[fmt].append(promedio)
    print(f"[{fmt.upper()}] Tamaño: {size} → Promedio: {promedio:.4f} s")

if __name__ == "__main__":
    # 🚀 Iniciar conexión y cargar datos
    cnx = get_connection()
    print("Connection established")
    data = get_data(cnx, "SELECT * FROM UN.VENTAS")
    df_full = pd.DataFrame(data, columns=[
        'ID_VENTA', 'FECHA_VENTA', 'ID_CLIENTE', 'ID_EMPLEADO',
        'ID_PRODUCTO', 'CANTIDAD', 'PRECIO_UNITARIO', 'DESCUENTO', 'FORMA_PAGO'
    ])
    cnx.close()
    print("Se obtuvieron todos los datos")

    # 🆕 Tamaños de muestra
    sizes = np.linspace(100, 1_000_000, num=30, dtype=int)
    sizes = sorted(list(set(sizes)))  # Eliminar posibles duplicados

    manager = Manager()
    results = manager.dict({fmt: manager.list() for fmt in formats})

    # ⚙️ Medición con multiprocessing
    for size in sizes:
        if size > len(df_full):
            print(f"❌ Saltando tamaño {size}, no hay suficientes datos.")
            continue

        df_sample = df_full.head(size)
        print(f"\n📦 Procesando tamaño: {size} registros")

        processes = []
        for fmt in formats:
            p = Process(target=medir_tiempo_promedio, args=(fmt, df_sample, size, results))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # ✅ Convertir resultados a diccionario normal
    results_dict = {fmt: list(results[fmt]) for fmt in formats}

    # 📄 Crear DataFrame con resultados
    df_resultados = pd.DataFrame({'Tamaño': sizes[:len(results_dict[formats[0]])]})
    for fmt in formats:
        df_resultados[fmt.upper()] = results_dict[fmt]

    # 💾 Exportar a CSV y JSON
    df_resultados.to_csv("output_files/resultados_serializacion.csv", index=False)
    df_resultados.to_json("output_files/resultados_serializacion.json", orient='records', indent=2)

    print("\n📂 Resultados guardados en output_files/resultados_serializacion.csv y .json")

    # 📊 Graficar
    plt.figure(figsize=(10, 6))
    for fmt in formats:
        plt.plot(df_resultados["Tamaño"], df_resultados[fmt.upper()], marker='o', label=fmt.upper())

    plt.xlabel("Cantidad de registros")
    plt.ylabel("Tiempo promedio de escritura (s)")
    plt.title("Comparación de formatos de serialización (multiprocessing, 10 repeticiones)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Guardar gráfica como imagen
    plt.savefig("output_files/tiempos_serializacion.png")
    plt.show()

    # 👀 Mostrar primeros resultados
    print("\n📊 Tabla de resultados:")
    print(df_resultados.head(10).to_string(index=False))