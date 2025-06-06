# Este script realiza el análisis de tiempo de ejecución de diferentes algoritmos de ordenamiento
# utilizando datos provenientes de un archivo JSON. También grafica los resultados y guarda 
# los datos procesados.
import json, time
import matplotlib.pyplot as plt
import threading
import json

# Diccionarios globales para almacenar resultados por algoritmo y los promedios
resultados={}
resultados_diccionarios={}



"""
    Clase que representa una fila (registro) individual de una venta extraída del archivo JSON.

    Atributos:
    - id_venta: Identificador único de la venta.
    - fecha_venta: Fecha en que se realizó la venta.
    - id_cliente: Identificador del cliente que realizó la compra.
    - id_empleado: Identificador del empleado que realizó la venta.
    - id_producto: Identificador del producto vendido.
    - cantidad: Cantidad del producto vendido.
    - precio_unitario: Precio unitario del producto.
    - descuento: Porcentaje o valor de descuento aplicado.
    - forma_pago: Método de pago utilizado (efectivo, tarjeta, etc.).
"""
class fila:
    def __init__(self, ID_VENTA, FECHA_VENTA, ID_CLIENTE, ID_EMPLEADO,ID_PRODUCTO, CANTIDAD, PRECIO_UNITARIO, DESCUENTO, FORMA_PAGO):
        self.id_cliente = ID_CLIENTE
        self.fecha_venta = FECHA_VENTA
        self.id_venta = ID_VENTA
        self.id_empleado = ID_EMPLEADO
        self.id_producto = ID_PRODUCTO
        self.cantidad = CANTIDAD
        self.precio_unitario = PRECIO_UNITARIO
        self.descuento = DESCUENTO
        self.forma_pago = FORMA_PAGO
        


# Carga del archivo JSON que contiene los registros de ventas. Cada línea representa una venta distinta.
# Los datos son almacenados como objetos de la clase 'fila' dentro de la lista 'lista_objetos_general'.
with open('/Users/bjose/Desktop/ED2-PF-202510/threads_sockets/ventas.json', 'r', encoding='utf-8') as file:
    lista_objetos_general=[]
    for line in file:

        venta = json.loads(line)

        id_venta = venta["ID_VENTA"]
        fecha = venta["FECHA_VENTA"]
        cliente = venta["ID_CLIENTE"]
        empleado = venta["ID_EMPLEADO"]
        producto = venta["ID_PRODUCTO"]
        cantidad = int(venta["CANTIDAD"])
        precio_unitario = float(venta["PRECIO_UNITARIO"])
        descuento = float(venta["DESCUENTO"])
        forma_pago = venta["FORMA_PAGO"]

        lista_objetos_general.append(fila(id_venta,fecha,cliente,empleado,producto,cantidad,precio_unitario,descuento,forma_pago))



# Función para calcular el promedio de tiempo de ejecución de un algoritmo a partir de múltiples ejecuciones.
"""
    Calcula el promedio de los tiempos de ejecución para cada cantidad de datos utilizados.

    Parámetros:
    - diccionario_total: Diccionario que contiene los tiempos de ejecución por experimento.
    - promedios: Diccionario donde se almacenarán los tiempos promedio por cantidad de elementos.
"""
def promedio(diccionario_total,promedios):
    for clave in diccionario_total[0].keys():
        suma = 0
        for j in range(5):
            suma += diccionario_total[j][clave]
        promedio = suma / 5
        promedios[clave] = promedio



# Función para graficar los tiempos de ejecución por cantidad de datos para un algoritmo específico.
"""
    Genera una gráfica de línea que representa el tiempo de ejecución de un algoritmo según el número de elementos procesados.

    Parámetros:
    - ordenamiento: Diccionario con los datos de tiempo.
    - nombre: Nombre del algoritmo (para el título y nombre del archivo de salida).
"""
def graficador(ordenamiento,nombre):
    # Ejes X e Y
    x = list(ordenamiento.keys())
    y = list(ordenamiento.values())

    # Crear gráfica
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, color='blue', linewidth=1)

    # Títulos y etiquetas
    plt.title("Tiempos de ejecución por número de elementos")
    plt.xlabel("Número de elementos")
    plt.ylabel("Tiempo (segundos)")

    # Mostrar gráfica
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{nombre}.png")
    


# Convierte una lista de objetos 'fila' a un archivo JSON legible por humanos para inspección o uso posterior.
"""
    Crea un archivo JSON a partir de una lista de objetos 'fila'.

    Parámetros:
    - lista_a_usar: Lista de objetos de tipo 'fila'.
    - metodo: Nombre del algoritmo que generó esta lista (se usa para el nombre del archivo).
"""
def rearmarJSON(lista_a_usar,metodo):
    ventas_dict_list = []
    for v in lista_a_usar:
        ventas_dict_list.append({"ID_VENTA": v.id_venta,"FECHA_VENTA": v.fecha_venta,"ID_CLIENTE": v.id_cliente,"ID_EMPLEADO": v.id_empleado,"ID_PRODUCTO": v.id_producto,"CANTIDAD": v.cantidad,"PRECIO_UNITARIO": v.precio_unitario,"DESCUENTO": v.descuento,"FORMA_PAGO": v.forma_pago})
    with open(f"{metodo}.json", "w") as archivo_json:
        json.dump(ventas_dict_list, archivo_json, indent=4)


        
"""
    Implementa el algoritmo de ordenamiento por inserción sobre una lista de objetos.

    El algoritmo recorre la lista desde el segundo elemento hacia adelante,
    y por cada elemento, lo compara con los anteriores para insertarlo en la posición correcta,
    asegurando que la sublista hasta ese índice esté ordenada.

    Se ordena la lista en base al atributo 'cantidad' de cada objeto.

    Parámetros:
        lista (list): Lista de objetos a ordenar, cada uno con un atributo 'cantidad'.

    Retorna:
        list: La misma lista pero ordenada según 'cantidad' en orden ascendente.
"""
def insertion(lista):
    for i in range(1, len(lista)):
        actual = lista[i]
        j = i - 1
        while j >= 0 and lista[j].cantidad > actual.cantidad:
            lista[j + 1] = lista[j]
            j -= 1
        lista[j + 1] = actual
    return lista
   


"""
    Mide el tiempo de ejecución del algoritmo de inserción para diferentes tamaños de listas.

    Para cada uno de 5 ciclos, va aumentando la cantidad de elementos desde 0 hasta 100000 en
    incrementos de 10000, ordenando esa sublista y guardando el tiempo transcurrido.

    Guarda los tiempos en un diccionario por tamaño y luego calcula el promedio por tamaño
    entre los 5 ciclos.

    Retorna:
        tuple: 
            - dict: Diccionario con los promedios de tiempo para cada tamaño de lista.
            - list: La última lista utilizada (lista de objetos ordenada).
"""
def tiempo_insercion():
    diccionario_total={}
    for j in range(5):
        diccionario={}
        r=0
        lista_objetos_tipo_fila=[]
        for i in range(0,50001,10000):
            while(r<=i):
                lista_objetos_tipo_fila.append(lista_objetos_general[r]) 
                r+=1
            lista_a_usar=lista_objetos_tipo_fila
            start = time.time()
            insertion(lista_a_usar)
            stop= time.time()
            diccionario[i]=stop-start
            
        diccionario_total[j]=diccionario
    Insercion={}
    promedio(diccionario_total,Insercion)
    print("In",len(Insercion))
    #graficador(Insercion,"Insercion")
    #rearmarJSON(lista_a_usar,"Insercion")
    return Insercion, lista_a_usar



"""
    Ejecuta la medición del tiempo total de ordenamiento por inserción sobre
    la lista completa 20 veces y calcula el promedio.

    Esto da un valor único promedio del tiempo para el algoritmo con la lista completa.

    Retorna:
        float: Tiempo promedio en segundos.
"""
def tiempo_insercion_total():
    diccionario={}
    for j in range(1):
        lista_a_usar=lista_objetos_general
        start = time.time()
        insertion(lista_a_usar)
        stop= time.time()
        diccionario[j]=stop-start
    valores = diccionario.values()
    promedio = sum(valores) / len(valores)
    print("INS")
    return promedio



"""
    Implementa el algoritmo de ordenamiento Bubble Sort (burbuja).

    Recorre repetidamente la lista comparando elementos adyacentes
    y los intercambia si están en el orden incorrecto, ordenando según
    el atributo 'cantidad' de los objetos.

    Parámetros:
        lista (list): Lista de objetos a ordenar.

    Retorna:
        list: Lista ordenada ascendentemente según 'cantidad'.
"""
def bubble_sort(lista):
    n = len(lista)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lista[j].cantidad > lista[j + 1].cantidad:
                lista[j], lista[j + 1] = lista[j + 1], lista[j]
    return lista



"""
    Mide el tiempo de ejecución del algoritmo Bubble Sort para sublistas crecientes,
    con cinco iteraciones completas de pruebas.

    Para cada iteración y tamaño, ordena la sublista y guarda el tiempo en diccionarios,
    luego calcula el promedio por tamaño.

    Retorna:
        tuple:
            - dict: Diccionario con tiempos promedio por tamaño.
            - list: Última sublista utilizada.
"""
def tiempo_bubble_sort():
    diccionario_total={}
    for j in range(5):
        diccionario={}
        r=0
        lista_objetos_tipo_fila=[]
        for i in range(1000):
            while(r<=i):
                lista_objetos_tipo_fila.append(lista_objetos_general[r]) 
                r+=1
            lista_a_usar=lista_objetos_tipo_fila
            start = time.time()
            bubble_sort(lista_a_usar)
            stop= time.time()
            diccionario[i]=stop-start
            
            
        diccionario_total[j]=diccionario
    Bubble_sort={}
    promedio(diccionario_total,Bubble_sort)
    print("BS",len(Bubble_sort))
    #graficador(Bubble_sort,"Bubble_Sort")
    #rearmarJSON(lista_a_usar,"Bubble_Sort")
    return Bubble_sort, lista_a_usar



"""
    Ejecuta el algoritmo Bubble Sort sobre la lista completa 20 veces,
    midiendo el tiempo total de ejecución y calculando el promedio.

    Retorna:
        float: Tiempo promedio en segundos.
"""
def tiempo_bubble_sort_total():
    diccionario={}
    for j in range(1):
        lista_a_usar=lista_objetos_general
        start = time.time()
        bubble_sort(lista_a_usar)
        stop= time.time()
        diccionario[j]=stop-start
    valores = diccionario.values()
    promedio = sum(valores) / len(valores)
    print("BS")
    return promedio



"""
    Implementa el algoritmo Counting Sort adaptado para ordenar objetos
    según el atributo 'cantidad'.

    Construye una lista de "cubetas" (buckets) para cada valor posible entre
    el mínimo y máximo de 'cantidad', luego concatena esos cubos en orden.

    Parámetros:
        lista (list): Lista de objetos a ordenar.

    Retorna:
        list: Lista ordenada por 'cantidad'.
"""
def counting_sort(lista):
    if not lista:
        return lista

    max_cantidad = max(obj.cantidad for obj in lista)
    min_cantidad = min(obj.cantidad for obj in lista)

    rango = max_cantidad - min_cantidad + 1
    count = [[] for _ in range(rango)]

    for obj in lista:
        count[obj.cantidad - min_cantidad].append(obj)

    resultado = []
    for bucket in count:
        resultado.extend(bucket)

    return resultado



"""
    Mide el tiempo de ejecución del Counting Sort para diferentes tamaños de sublistas,
    repitiendo 5 veces para obtener un promedio de tiempos.

    Retorna:
        tuple:
            - dict: Tiempos promedio por tamaño.
            - list: Última sublista usada.
"""
def tiempo_counting_sort():
    diccionario_total={}
    for j in range(5):
        diccionario={}
        r=0
        lista_objetos_tipo_fila=[]
        for i in range(0,50001,10000):
            while(r<=i):
                lista_objetos_tipo_fila.append(lista_objetos_general[r]) 
                r+=1
            lista_a_usar=lista_objetos_tipo_fila
            start = time.time()
            counting_sort(lista_a_usar)
            stop= time.time()
            diccionario[i]=stop-start
            
        diccionario_total[j]=diccionario
    Counting_sort={}
    promedio(diccionario_total,Counting_sort)
    print("CS",len(Counting_sort))
    #graficador(Counting_sort,"Counting_sort")
    #rearmarJSON(lista_a_usar,"CountingSort")
    return Counting_sort, lista_a_usar



"""
    Ejecuta el Counting Sort sobre la lista completa 20 veces y calcula
    el tiempo promedio de ejecución.

    Retorna:
        float: Tiempo promedio en segundos.
"""
def tiempo_counting_sort_total():
    diccionario={}
    for j in range(1):
        lista_a_usar=lista_objetos_general
        start = time.time()
        counting_sort(lista_a_usar)
        stop= time.time()
        diccionario[j]=stop-start
    valores = diccionario.values()
    promedio = sum(valores) / len(valores)
    print("CS")
    return promedio



"""
    Función auxiliar para Quick Sort.

    Elige el pivote como el elemento medio y reorganiza el array para que todos
    los elementos menores al pivote estén a la izquierda y los mayores a la derecha.

    Parámetros:
        array (list): Lista de objetos a ordenar.
        low (int): Índice bajo del subarreglo.
        high (int): Índice alto del subarreglo.

    Retorna:
        tuple: índices (i, j) para los límites de partición.
"""
def partition(array, low, high):
    pivot = array[(low + high) // 2].cantidad
    i = low
    j = high
    while i <= j:
        while array[i].cantidad < pivot:
            i += 1
        while array[j].cantidad > pivot:
            j -= 1
        if i <= j:
            array[i], array[j] = array[j], array[i]
            i += 1
            j -= 1
    return i, j  # Devolvemos ambos índices para saber los límites


"""
    Implementa Quick Sort de forma iterativa para ordenar una lista de objetos
    según su atributo 'cantidad'.

    Usa una pila para manejar los límites de las particiones pendientes y
    la función partition para dividir.

    Parámetros:
        array (list): Lista de objetos a ordenar.

    No retorna, ordena la lista in-place.
"""
def quick_sort_iterative(array):
    stack = [(0, len(array) - 1)]
    while stack:
        low, high = stack.pop()
        if low < high:
            i, j = partition(array, low, high)
            if low < j:
                stack.append((low, j))
            if i < high:
                stack.append((i, high))

"""
    Mide el tiempo de ejecución del Quick Sort para sublistas de tamaño creciente,
    repitiendo 5 veces y calculando promedios para cada tamaño.

    Retorna:
        tuple:
            - dict: Tiempos promedio por tamaño.
            - list: Última sublista usada.
"""           
def tiempo_quick_sort():
    diccionario_total={}
    for j in range(5):
        diccionario={}
        r=0
        lista_objetos_tipo_fila=[]
        for i in range(0,50001,10000):
            while(r<=i):
                lista_objetos_tipo_fila.append(lista_objetos_general[r])
                r+=1
            lista_a_usar=lista_objetos_tipo_fila
            start = time.time()
            quick_sort_iterative(lista_a_usar)
            stop= time.time()
            diccionario[i]=stop-start
            
        diccionario_total[j]=diccionario
    Quick_sort={}
    promedio(diccionario_total,Quick_sort)
    print("QS",len(Quick_sort))
    #graficador(Quick_sort,"Quick_sort")
    #rearmarJSON(lista_a_usar,"QuickSort")
    return Quick_sort, lista_a_usar



"""
    Ejecuta Quick Sort sobre la lista completa 20 veces y calcula
    el promedio de los tiempos medidos.

    Retorna:
        float: Tiempo promedio en segundos.
"""
def tiempo_quick_sort_total():
    diccionario={}
    for j in range(1):
        lista_a_usar=lista_objetos_general
        start = time.time()
        quick_sort_iterative(lista_a_usar)
        stop= time.time()
        diccionario[j]=stop-start
    valores = diccionario.values()
    promedio = sum(valores) / len(valores)
    print("QS")
    return promedio



if __name__ == "__main__":
    import queue

    resultados = queue.Queue()

    def wrapper(func):
        def inner():
            res = func()
            resultados.put((func.__name__, res))
        return inner

    hilo1 = threading.Thread(target=wrapper(tiempo_counting_sort))
    hilo2 = threading.Thread(target=wrapper(tiempo_insercion))
    hilo3 = threading.Thread(target=wrapper(tiempo_quick_sort))
    hilo4 = threading.Thread(target=wrapper(tiempo_bubble_sort))

    hilo1.start()
    hilo2.start()
    hilo3.start()
    hilo4.start()

    hilo1.join()
    hilo2.join()
    hilo3.join()
    hilo4.join()

    # Recibir resultados y graficar en hilo principal
    while not resultados.empty():
        nombre_func, (datos, lista_objetos) = resultados.get()
        graficador(datos, nombre_func)
        rearmarJSON(lista_objetos, nombre_func)
        resultados_diccionarios[nombre_func] = datos

    print("Todos los métodos han terminado.")

    json_completo2 = {
            "datos_por_algoritmo": resultados_diccionarios,
    }

with open("resultado2.json", "w", encoding="utf-8") as f2:
    json.dump(json_completo2, f2, indent=4)

def graficar_tiempos_promedio(ins, bubble, counting, quick):
    algoritmos = ['Inserción', 'Bubble Sort', 'Counting Sort', 'Quick Sort']
    tiempos = [ins, bubble, counting, quick]

    plt.figure(figsize=(8,6))
    plt.bar(algoritmos, tiempos, color=['blue', 'orange', 'green', 'red'])

    plt.title('Tiempo promedio de ejecución por algoritmo')
    plt.xlabel('Algoritmo')
    plt.ylabel('Tiempo (segundos)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, tiempo in enumerate(tiempos):
        plt.text(i, tiempo + max(tiempos)*0.01, f'{tiempo:.4f}', ha='center')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GraficoGeneral.png")

# Variables donde se guardarán los promedios
prom_ins = [0]
prom_bubble = [0]
prom_counting = [0]
prom_quick = [0]

# Funciones wrapper para cada hilo
def hilo_insercion():
    prom_ins[0] = tiempo_insercion_total()

def hilo_bubble():
    prom_bubble[0] = tiempo_bubble_sort_total()

def hilo_counting():
    prom_counting[0] = tiempo_counting_sort_total()

def hilo_quick():
    prom_quick[0] = tiempo_quick_sort_total()


if __name__ == "__main__":
    # Crear los hilos
    t1 = threading.Thread(target=hilo_insercion)
    #t2 = threading.Thread(target=hilo_bubble)
    t3 = threading.Thread(target=hilo_counting)
    t4 = threading.Thread(target=hilo_quick)

    # Iniciar los hilos
    t1.start()
    #t2.start()
    t3.start()
    t4.start()

    # Esperar a que todos terminen
    t1.join()
    #t2.join()
    t3.join()
    t4.join()

    # Usar los valores retornados
    graficar_tiempos_promedio(prom_ins[0], prom_bubble[0], prom_counting[0], prom_quick[0])
    resultados = {
        "Insercion": prom_ins,
        "Burbuja": prom_bubble,
        "Counting": prom_counting,
        "QuickSort": prom_quick
    }

    print("Todos los métodos terminaron y gráfica creada.")

    json_completo1 = {
            "datos_promedios": resultados
    }

with open("resultado1.json", "w", encoding="utf-8") as f1:
    json.dump(json_completo1, f1, indent=4)


print("¡Archivos JSON guardados!")

    