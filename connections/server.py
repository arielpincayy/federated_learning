import asyncio
import os
import time
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
    Entrada: dirección del servidor (host:port), duración máxima de inactividad en segundos.
    Salida: lista de direcciones únicas (host:port) registradas por los clientes.
    
    Lógica: El servidor se cerrará automáticamente si pasa 'duration' segundos 
            sin que ningún nodo nuevo se conecte (Timeout dinámico tras el último nodo).
    """
    host, port = get_ipport(addr)
    print(f"[TEMP SERVER] Iniciando en ws://{host}:{port}...")
    print(f"[TEMP SERVER] El servidor cerrará tras {duration}s de inactividad desde el último nodo.")

    addrs = []
    
    # Registramos el timestamp actual como el punto de inicio de la espera
    ultimo_registro_time = time.time()

    async def wrapper(websocket):
        nonlocal ultimo_registro_time
        try:
            message = await websocket.recv()
            if isinstance(message, str) and message not in addrs:
                addrs.append(message)
                print(f"[TEMP SERVER] Nodo registrado: {message}")
                
                # ¡AQUÍ REINICIAMOS EL TEMPORIZADOR!
                # Al actualizar el timestamp, obligamos al bucle principal a volver a esperar 'duration' segundos
                ultimo_registro_time = time.time()
                
                await websocket.send(ACK_REGISTERED)
        except websockets.exceptions.ConnectionClosed:
            pass

    # Iniciamos el servidor de websockets
    server = await websockets.serve(wrapper, host, port)
    
    try:
        # Bucle de monitoreo dinámico
        while True:
            await asyncio.sleep(1) # Revisamos las condiciones una vez por segundo
            
            tiempo_inactivo = time.time() - ultimo_registro_time
            if tiempo_inactivo >= duration:
                print(f"[TEMP SERVER] Se alcanzó el límite de {duration}s sin nuevas conexiones.")
                break
    finally:
        # Garantizamos que el servidor se cierre limpiamente al salir del bucle
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
    file_path: str = None,
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
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    with open(file_path, "wb") as f:
                        f.write(message)
                    print(f"[FILE SAVED] {file_path}")
                    await websocket.send("File received successfully")
                    return file_path
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