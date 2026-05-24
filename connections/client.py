import asyncio
import websockets
from config import ACK_IDENTIFIED
from utils import get_ipport


async def send(server_addr: str, message: str):
    """
    Entrada: dirección del servidor (host:port), mensaje de texto a enviar.
    Salida: ninguna (imprime respuesta del servidor o error).
    
    Intenta conectarse y enviar el mensaje hasta 3 veces antes de fallar.
    """
    if not isinstance(message, str):
        print(f"[ERROR] send() espera str. Recibido: {type(message)}")
        return

    host, port = get_ipport(server_addr)
    uri = f"ws://{host}:{port}"
    
    max_intentos = 3
    delay_entre_intentos = 2  # Segundos a esperar antes de reintentar

    for intento in range(1, max_intentos + 1):
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(message)
                response = await websocket.recv()
                print(f"[SERVER RESPONSE] {response}")
                return  # Éxito: salimos de la función inmediatamente
                
        except Exception as e:
            print(f"[ERROR EN SEND] Intento {intento}/{max_intentos} falló: {e}")
            
            # Si aún nos quedan intentos, esperamos un momento antes de volver a probar
            if intento < max_intentos:
                print(f"[SEND] Reintentando en {delay_entre_intentos} segundos...")
                await asyncio.sleep(delay_entre_intentos)
            else:
                print(f"[CRITICAL] No se pudo conectar con {uri} tras {max_intentos} intentos.")

async def send_identified(node_addr: str, server_addr: str, message: str | bytes):
    """
    Entrada: addr propio del nodo (host:port) usado como identificación,
             dirección del servidor (host:port), mensaje de texto o bytes a enviar.
    Salida: ninguna (envía identificación primero, luego el mensaje).
    """
    host, port = get_ipport(server_addr)
    uri = f"ws://{host}:{port}"
    try:
        async with websockets.connect(uri) as websocket:
            # Identificación
            await websocket.send(node_addr)
            ack = await websocket.recv()
            if ack != ACK_IDENTIFIED:
                print(f"[ERROR] Identificación rechazada: {ack}")
                return

            # Payload
            await websocket.send(message)
            response = await websocket.recv()
            print(f"[SERVER RESPONSE] {response}")
    except Exception as e:
        print(f"[ERROR EN SEND_IDENTIFIED] {e}")


async def send_file_identified(node_addr: str, server_addr: str, file_path: str):
    """
    Entrada: addr propio del nodo (host:port) usado como identificación,
             dirección del servidor (host:port), ruta del archivo a enviar.
    Salida: ninguna (envía identificación, luego el archivo como bytes).
    """
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
    except FileNotFoundError:
        print(f"[ERROR] Archivo no encontrado: {file_path}")
        return
    except Exception as e:
        print(f"[ERROR AL LEER ARCHIVO] {e}")
        return

    host, port = get_ipport(server_addr)
    uri = f"ws://{host}:{port}"
    try:
        async with websockets.connect(uri) as websocket:
            # Identificación
            await websocket.send(node_addr)
            ack = await websocket.recv()
            if ack != ACK_IDENTIFIED:
                print(f"[ERROR] Identificación rechazada: {ack}")
                return

            # Archivo
            print(f"[CLIENT] Transmitiendo '{file_path}' ({len(file_data)} bytes)...")
            await websocket.send(file_data)
            response = await websocket.recv()
            print(f"[SERVER RESPONSE] {response}")
    except Exception as e:
        print(f"[ERROR EN SEND_FILE_IDENTIFIED] {e}")


async def send_file_to_nodes(node_addrs: list[str], filepath: str, delay: float):
    """
    Entrada: lista de direcciones de los nodos (['ip1:port1', 'ip2:port2'...]), 
             ruta del archivo a enviar, tiempo máximo de espera global en segundos.
    Salida: ninguna (conecta a cada nodo y le envía el archivo en paralelo).
    """
    print(f"[SERVER] Iniciando distribución de archivo {filepath} a {len(node_addrs)} nodo(s)...")

    # Leemos el archivo en memoria una sola vez para optimizar
    with open(filepath, "rb") as f:
        data = f.read()

    async def send_to_single_node(addr: str):
        try:
            # Nos conectamos activamente al nodo receptor
            async with websockets.connect(f"ws://{addr}") as websocket:
                await websocket.send(data)
                response = await websocket.recv()
                print(f"[SERVER] Confirmación desde {addr}: {response}")
                return True
        except (websockets.exceptions.ConnectionClosed, OSError, asyncio.TimeoutError):
            print(f"[SERVER] Error o timeout al intentar conectar con el nodo {addr}")
            return False

    # Creamos las tareas para disparar los envíos de forma simultánea
    tasks = [send_to_single_node(addr) for addr in node_addrs]
    
    try:
        # Se ejecutan todas en paralelo respetando el timeout global
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=delay)
        served = sum(1 for r in results if r is True)
    except asyncio.TimeoutError:
        print("[SERVER] Timeout global alcanzado. Algunas conexiones quedaron incompletas.")
        # Intentamos recuperar los resultados completados si gather ya había guardado algunos
        served = "Incompleto (Timeout)"

    print(f"[SERVER] Archivo entregado con éxito a {served}/{len(node_addrs)} nodo(s).")


async def send_message_to_nodes(node_addrs: list[str], message: str, delay: float):
    """
    Entrada: lista de direcciones de los nodos (['ip1:port1', 'ip2:port2'...]), 
             mensaje (str) a enviar, tiempo máximo de espera global en segundos.
    Salida: ninguna (conecta a cada nodo y le envía el mensaje en paralelo).
    """
    print(f"[SERVER] Iniciando distribución de mensaje a {len(node_addrs)} nodo(s)...")

    async def send_to_single_node(addr: str):
        try:
            # Nos conectamos activamente al nodo receptor
            async with websockets.connect(f"ws://{addr}") as websocket:
                await websocket.send(message)
                response = await websocket.recv()
                print(f"[SERVER] Confirmación desde {addr}: {response}")
                return True
        except (websockets.exceptions.ConnectionClosed, OSError, asyncio.TimeoutError):
            print(f"[SERVER] Error o timeout al intentar conectar con el nodo {addr}")
            return False

    # Creamos las tareas para todos los nodos
    tasks = [send_to_single_node(addr) for addr in node_addrs]
    
    try:
        # Enviamos a todos a la vez de forma asíncrona
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=delay)
        served = sum(1 for r in results if r is True)
    except asyncio.TimeoutError:
        print("[SERVER] Timeout global alcanzado. Algunas conexiones quedaron incompletas.")
        served = "Incompleto (Timeout)"

    print(f"[SERVER] Mensaje entregado con éxito a {served}/{len(node_addrs)} nodo(s).")