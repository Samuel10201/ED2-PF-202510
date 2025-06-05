import socket
import pickle
import pandas as pd
import io
import time
import pyarrow as pa
from fastavro import writer, parse_schema
import datetime
from decimal import Decimal
import tempfile
import multiprocessing as mp
import psutil
import os

def serialize(df, fmt):
    df = df.copy()

    if fmt in ['json', 'csv']:
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (datetime.date, datetime.datetime))).any():
                df[col] = df[col].astype(str)

    if fmt == 'json':
        with io.StringIO() as buf:
            df.to_json(buf, orient='records', lines=True)
            _ = buf.getvalue()

    elif fmt == 'csv':
        with io.StringIO() as buf:
            df.to_csv(buf, index=False)
            _ = buf.getvalue()

    elif fmt == 'parquet':
        table = pa.Table.from_pandas(df)
        with pa.BufferOutputStream() as buf:
            pa.write_table(table, buf)

    elif fmt == 'avro':
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (datetime.date, datetime.datetime))).any():
                df[col] = df[col].astype(str)
            elif df[col].apply(lambda x: isinstance(x, Decimal)).any():
                df[col] = df[col].astype(float)

        records = df.to_dict(orient='records')
        schema = {
            "type": "record",
            "name": "Venta",
            "fields": []
        }

        avro_types = {
            'int64': 'long',
            'float64': 'double',
            'object': 'string',
            'bool': 'boolean'
        }

        for col in df.columns:
            dtype = str(df[col].dtype)
            avro_type = avro_types.get(dtype, 'string')
            schema["fields"].append({
                "name": col,
                "type": ["null", avro_type],
                "default": None
            })

        with tempfile.NamedTemporaryFile(delete=True, suffix=".avro") as temp_file:
            writer(temp_file, parse_schema(schema), records)

    else:
        raise ValueError("Formato no soportado")

def worker_serialize(fmt, df, queue):
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # reset

    tiempos = []
    mem_usages = []

    for _ in range(10):
        start = time.time()
        serialize(df, fmt)
        tiempos.append(time.time() - start)
        mem_usages.append(process.memory_info().rss)  # Resident Set Size (RAM)

    avg_time = sum(tiempos) / len(tiempos)
    max_mem = max(mem_usages) / (1024 * 1024)  # MB
    cpu_percent = process.cpu_percent(interval=0.1)

    queue.put((fmt, avg_time, max_mem, cpu_percent))

def medir_tiempos(df):
    formatos = ['json', 'csv', 'parquet', 'avro']
    queue = mp.Queue()
    procesos = []

    inicio_total = time.time()

    for fmt in formatos:
        p = mp.Process(target=worker_serialize, args=(fmt, df, queue))
        procesos.append(p)
        p.start()

    for p in procesos:
        p.join()

    tiempo_total = time.time() - inicio_total

    resultados = {}
    while not queue.empty():
        fmt, avg_time, max_mem, cpu_percent = queue.get()
        resultados[fmt] = {
            'tiempo_promedio': avg_time,
            'memoria_MB': max_mem,
            'cpu_%': cpu_percent
        }

    resultados['tiempo_total'] = tiempo_total
    return resultados

if __name__ == "__main__":
    HOST = 'localhost'
    PORT = 9999

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("[🟡] Conectado al servidor. Esperando datos...")

        length = int.from_bytes(s.recv(8), 'big')
        received = b''
        while len(received) < length:
            received += s.recv(min(4096, length - len(received)))
        df = pickle.loads(received)

        print(f"[📦] Datos recibidos: {len(df)} registros. Procesando...\n")

        resultados = medir_tiempos(df)

        print("📊 Resultados de serialización (promedios):")
        for fmt in ['json', 'csv', 'parquet', 'avro']:
            res = resultados[fmt]
            print(f"🔸 {fmt.upper()}: {res['tiempo_promedio']:.4f}s | "
                  f"{res['memoria_MB']:.2f} MB | CPU: {res['cpu_%']:.1f}%")

        print(f"\n⏱️ Tiempo total (paralelo): {resultados['tiempo_total']:.4f}s")

        result_bytes = pickle.dumps(resultados)
        s.sendall(len(result_bytes).to_bytes(8, 'big'))
        s.sendall(result_bytes)

        print("[✅] Resultados enviados al servidor.")

