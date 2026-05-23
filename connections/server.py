import asyncio
import os
import websockets

from config import LISTENER_DURATION, RECEIVED_FILES_PATH, RECEIVED_MODEL_FILENAME, ACK_IDENTIFIED, ACK_REGISTERED, ACK_FILE_SUCCESS, ACK_MESSAGE_SUCCESS
from utils import get_ipport


async def _save_file(data: bytes, save_path: str, filename: str = RECEIVED_MODEL_FILENAME) -> str:
    """
    Entrada: datos binarios, directorio destino, nombre de archivo opcional.
    Salida: ruta donde se guardó el archivo.
    """
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    with open(filepath, "wb") as f:
        f.write(data)
    print(f"[FILE SAVED] {filepath}")
    return filepath


async def listener_ips(addr: str, duration: int = LISTENER_DURATION) -> list[str]:
    """
    Entrada: dirección del servidor (host:port), duración en segundos.
    Salida: lista de direcciones únicas (host:port) tal como las enviaron los clientes.
    """
    host, port = get_ipport(addr)
    print(f"[TEMP SERVER] Listening on ws://{host}:{port} for {duration}s")

    addrs = []

    async def wrapper(websocket):
        try:
            message = await websocket.recv()
            if isinstance(message, str) and message not in addrs:
                addrs.append(message)
                print(f"[TEMP SERVER] Nodo registrado: {message}")
                await websocket.send(ACK_REGISTERED)
        except websockets.exceptions.ConnectionClosed:
            pass

    server = await websockets.serve(wrapper, host, port)
    try:
        await asyncio.sleep(duration)
    finally:
        server.close()
        await server.wait_closed()
        print("[TEMP SERVER] Closed")

    return addrs


async def listener_nodes(
    addr: str,
    nodes: dict[str, str],
    delay: float,
    save_path: str = None,
) -> dict[str, list]:
    """
    Entrada: dirección propia (host:port), dict de {addr -> identificador},
             tiempo máximo de escucha en segundos, directorio para guardar archivos.
    Salida: dict de {identificador -> lista} con str recibidos o rutas de archivos guardados.
            Cierra en cuanto todos los nodos esperados han enviado modelo y métricas,
            o cuando se agota el delay. Conexiones no identificadas se ignoran.
    """
    host, port = get_ipport(addr)
    n_expected = len(nodes)
    print(f"[SERVER] Listening on ws://{host}:{port} "
          f"(esperando {n_expected} nodo(s), max {delay}s)...")

    results: dict[str, list] = {}
    # Rastreamos qué nodos ya enviaron modelo Y métricas
    completed: set[str] = set()
    done_event = asyncio.Event()

    async def wrapper(websocket):
        try:
            ident = await websocket.recv()
            if not isinstance(ident, str) or ident not in nodes:
                print(f"[SERVER] Identificación desconocida ignorada: {ident!r}")
                return

            node_id = nodes[ident]
            if node_id not in results:
                results[node_id] = []

            await websocket.send(ACK_IDENTIFIED)

            async for message in websocket:
                if isinstance(message, bytes):
                    path = save_path or RECEIVED_FILES_PATH
                    filename = f"{node_id}_model.pt"
                    filepath = await _save_file(message, path, filename)
                    results[node_id].append(filepath)
                    await websocket.send(ACK_FILE_SUCCESS)
                else:
                    print(f"[SERVER] [{node_id}] str recibido: {message}")
                    results[node_id].append(message)
                    await websocket.send(ACK_MESSAGE_SUCCESS)

                # Consideramos un nodo completo cuando tiene modelo (.pt) y métricas (str JSON)
                has_model   = any(isinstance(i, str) and i.endswith(".pt") for i in results[node_id])
                has_metrics = any(isinstance(i, str) and not i.endswith(".pt") for i in results[node_id])
                if has_model and has_metrics:
                    completed.add(node_id)
                    print(f"[SERVER] [{node_id}] completado ({len(completed)}/{n_expected})")
                    if len(completed) >= n_expected:
                        done_event.set()

        except websockets.exceptions.ConnectionClosed:
            pass

    async with websockets.serve(wrapper, host, port):
        try:
            await asyncio.wait_for(done_event.wait(), timeout=delay)
            print(f"[SERVER] Todos los nodos completados.")
        except asyncio.TimeoutError:
            print(f"[SERVER] Timeout: {len(completed)}/{n_expected} nodos completados.")

    return results


async def listener_server(
    server_addr: str,
    delay: float,
    save_path: str = None,
    retry_interval: float = 1.0,
) -> str | None:
    """
    Entrada: dirección del servidor central (host:port), tiempo máximo de espera en segundos,
             directorio para guardar archivos recibidos, intervalo de reintento en segundos.
    Salida: str recibido, ruta del archivo guardado si llegaron bytes,
            o None si se agotó el tiempo total sin recibir nada.
    """
    ip, port = get_ipport(server_addr)
    uri = f"ws://{ip}:{port}"
    print(f"[CLIENT] Esperando modelo de {uri} (max {delay}s)...")

    deadline = asyncio.get_event_loop().time() + delay

    while asyncio.get_event_loop().time() < deadline:
        try:
            remaining = deadline - asyncio.get_event_loop().time()
            async with websockets.connect(uri) as websocket:
                message = await asyncio.wait_for(websocket.recv(), timeout=remaining)

                if isinstance(message, bytes):
                    path = save_path or "received_files"
                    os.makedirs(path, exist_ok=True)
                    filepath = os.path.join(path, "received_model.pt")
                    with open(filepath, "wb") as f:
                        f.write(message)
                    print(f"[FILE SAVED] {filepath}")
                    await websocket.send("File received successfully")
                    return filepath
                else:
                    print(f"[CLIENT] Mensaje recibido: {message}")
                    await websocket.send("Message received")
                    return message

        except (ConnectionRefusedError, OSError):
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(retry_interval)
        except asyncio.TimeoutError:
            break
        except websockets.exceptions.ConnectionClosed:
            await asyncio.sleep(retry_interval)
        except Exception as e:
            print(f"[CLIENT] Error: {e}")
            break

    print(f"[CLIENT] Tiempo agotado: no se recibió modelo de {uri}.")
    return None