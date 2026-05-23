import asyncio
import websockets
from config import ACK_IDENTIFIED
from utils import get_ipport


async def send(server_addr: str, message: str):
    """
    Entrada: dirección del servidor (host:port), mensaje de texto a enviar.
    Salida: ninguna (imprime respuesta del servidor o error).
    """
    if not isinstance(message, str):
        print(f"[ERROR] send() espera str. Recibido: {type(message)}")
        return

    host, port = get_ipport(server_addr)
    uri = f"ws://{host}:{port}"
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(message)
            response = await websocket.recv()
            print(f"[SERVER RESPONSE] {response}")
    except Exception as e:
        print(f"[ERROR EN SEND] {e}")


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


async def server_file(addr: str, filepath: str, n_clients: int, delay: float):
    """
    Entrada: dirección propia (host:port), ruta del archivo a servir,
             número de clientes esperados, tiempo máximo en segundos.
    Salida: ninguna (envía el archivo a cada cliente que se conecte).
    """
    host, port = get_ipport(addr)
    print(f"[SERVER] Sirviendo {filepath} en ws://{host}:{port} para {n_clients} cliente(s)...")

    with open(filepath, "rb") as f:
        data = f.read()

    served = 0
    event = asyncio.Event()

    async def wrapper(websocket):
        nonlocal served
        try:
            await websocket.send(data)
            response = await websocket.recv()
            print(f"[SERVER] Confirmación de nodo: {response}")
            served += 1
            if served >= n_clients:
                event.set()
        except websockets.exceptions.ConnectionClosed:
            pass

    async with websockets.serve(wrapper, host, port):
        try:
            await asyncio.wait_for(event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            print(f"[SERVER] Timeout: {served}/{n_clients} nodos recibieron el modelo.")

    print(f"[SERVER] Modelo entregado a {served}/{n_clients} nodo(s).")