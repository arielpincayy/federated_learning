import asyncio
import os
import time
import websockets

from config import LISTENER_DURATION, RECEIVED_FILES_PATH, ACK_IDENTIFIED, ACK_REGISTERED, ACK_FILE_SUCCESS, ACK_MESSAGE_SUCCESS, SLEEP_INTERVAL
from utils import get_ipport, _save_file
from logging_config import get_logger
from network_metrics import NetworkMetricsContext

logger = get_logger(__name__)


async def listener_ips(addr: str, duration: int = LISTENER_DURATION) -> list[str]:
    """
    Entrada: dirección del servidor (host:port), duración máxima de inactividad en segundos.
    Salida: lista de direcciones únicas (host:port) registradas por los clientes.

    Cierra automáticamente tras `duration` segundos de inactividad. Cada nueva 
    conexión reinicia el temporizador, esperando `duration` segundos adicionales.
    """
    host, port = get_ipport(addr)
    logger.info(f"[TEMP SERVER] Iniciando en ws://{host}:{port}...")
    logger.info(f"[TEMP SERVER] El servidor cerrará si pasan {duration}s sin nuevos nodos.")

    addrs = []
    # Inicializamos el marcador de tiempo al arrancar el servidor
    ultimo_registro_time = time.time()

    async def wrapper(websocket):
        nonlocal ultimo_registro_time
        try:
            message = await websocket.recv()
            if isinstance(message, str):
                # Actualizamos el timestamp inmediatamente al recibir la señal
                ultimo_registro_time = time.time()
                
                if message not in addrs:
                    addrs.append(message)
                    logger.info(f"[TEMP SERVER] Nodo registrado: {message}. Temporizador reiniciado (+{duration}s).")
                else:
                    logger.info(f"[TEMP SERVER] Nodo {message} ya estaba registrado. Temporizador reiniciado (+{duration}s).")
                
                await websocket.send(ACK_REGISTERED)
        except websockets.exceptions.ConnectionClosed:
            pass

    # Iniciamos el servidor de websockets
    server = await websockets.serve(wrapper, host, port)
    
    try:
        while True:
            await asyncio.sleep(0.5)  # Ajusta este intervalo según tu SLEEP_INTERVAL
            
            tiempo_inactivo = time.time() - ultimo_registro_time
            if tiempo_inactivo >= duration:
                logger.info(f"[TEMP SERVER] Se alcanzó el límite de {duration}s de inactividad total. Cerrando...")
                break
    finally:
        # Garantizamos el cierre correcto del servidor y la liberación del puerto
        server.close()
        await server.wait_closed()
        logger.info("[TEMP SERVER] Servidor temporal cerrado exitosamente.")

    return addrs


async def listener_nodes(addr: str, nodes: dict[str, str], delay: float, save_path: str = None, message_only: str = False) -> dict[str, list]:
    """
    Entrada: dirección propia (host:port), dict de {addr -> identificador},
             tiempo máximo de escucha en segundos, directorio para guardar archivos.
    Salida: dict de {identificador -> lista} con str recibidos o rutas de archivos guardados.
            Incluye métricas de red en cada ítem como dict cuando se reciben archivos.
            Cierra en cuanto todos los nodos completan o se agota el delay.
    """
    host, port = get_ipport(addr)
    n_expected = len(nodes)
    logger.info(f"[SERVER] Listening on ws://{host}:{port} "
          f"(esperando {n_expected} nodo(s), max {delay}s)...")

    # Lookup por host (IP) ignorando puerto
    nodes_by_host: dict[str, str] = {get_ipport(a)[0]: nid for a, nid in nodes.items()}

    results: dict[str, list] = {}
    node_metrics: dict[str, dict] = {}  # Almacena métricas de red por nodo
    completed: set[str] = set()
    done_event = asyncio.Event()

    async def wrapper(websocket):
        try:
            ident = await websocket.recv()
            if not isinstance(ident, str):
                logger.error(f"[SERVER] Identificación inválida ignorada: {ident!r}")
                return

            try:
                ident_host, _ = get_ipport(ident)
            except Exception:
                logger.error(f"[SERVER] Identificación mal formada ignorada: {ident!r}")
                return

            if ident_host not in nodes_by_host:
                logger.error(f"[SERVER] Host desconocido ignorado: {ident_host!r}")
                return

            node_id = nodes_by_host[ident_host]
            if node_id not in results:
                results[node_id] = []
                node_metrics[node_id] = {}

            await websocket.send(ACK_IDENTIFIED)

            # Iniciar monitoreo de red para este nodo
            with NetworkMetricsContext() as metrics:
                async for message in websocket:
                    if isinstance(message, bytes):
                        path = save_path or RECEIVED_FILES_PATH
                        filename = f"{node_id}_model.pt"
                        filepath = await _save_file(message, path, filename)
                        results[node_id].append(filepath)
                        # Registrar bytes del modelo recibidos
                        metrics.set_model_bytes_transferred(bytes_rx=len(message))
                        await websocket.send(ACK_FILE_SUCCESS)
                    else:
                        logger.info(f"[SERVER] [{node_id}] str recibido: {message}")
                        results[node_id].append(message)
                        await websocket.send(ACK_MESSAGE_SUCCESS)
                    if message_only:
                        completed.add(node_id)
                        if len(completed) >= n_expected:
                            done_event.set()
                    else:
                        has_model   = any(isinstance(i, str) and i.endswith(".pt") for i in results[node_id])
                        has_metrics = any(isinstance(i, str) and not i.endswith(".pt") for i in results[node_id])
                        if has_model and has_metrics:
                            # Guardar métricas de red cuando se completa el nodo
                            net_metrics = metrics.end_monitoring()
                            node_metrics[node_id] = net_metrics
                            completed.add(node_id)
                            logger.info(f"[SERVER] [{node_id}] completado ({len(completed)}/{n_expected}) - "
                                       f"Red: TX={net_metrics.get('net_bytes_tx_system', 0)}B, "
                                       f"RX={net_metrics.get('net_bytes_rx_system', 0)}B, "
                                       f"BW={net_metrics.get('net_throughput_kbps', 0):.2f}kbps")
                            if len(completed) >= n_expected:
                                done_event.set()

        except websockets.exceptions.ConnectionClosed:
            pass

    async with websockets.serve(wrapper, host, port):
        try:
            await asyncio.wait_for(done_event.wait(), timeout=delay)
            logger.info(f"[SERVER] Todos los nodos completados.")
        except asyncio.TimeoutError:
            logger.warning(f"[SERVER] Timeout: {len(completed)}/{n_expected} nodos completados.")

    # Añadir métricas de red a los resultados
    for node_id, metrics in node_metrics.items():
        if results[node_id]:
            results[node_id].append(metrics)

    return results


async def listener_server(
    listen_addr: str, 
    delay: float, 
    file_path: str = None
) -> str | bytes | None:
    """
    Entrada: dirección local donde el nodo escuchará (host:port o ip:port), 
             tiempo máximo de espera en segundos,
             ruta donde guardar el archivo si se reciben bytes (None = no guardar).
    Salida: tupla (resultado, métricas_red) o None si se agota el tiempo sin recibir nada.
    """
    host, port = get_ipport(listen_addr)
    logger.info(f"[NODE] Escuchando en ws://{host}:{port} esperando al servidor central (max {delay}s)...")

    # Variables para capturar lo que recibamos y un evento para romper la espera
    result = None
    net_metrics = {}
    stop_event = asyncio.Event()

    async def handler(websocket):
        nonlocal result, net_metrics
        try:
            with NetworkMetricsContext() as metrics:
                # Esperamos a que el servidor central nos envíe la información
                message = await websocket.recv()

                if isinstance(message, bytes):
                    metrics.set_model_bytes_transferred(bytes_rx=len(message))
                    if file_path:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        # Aseguramos que el directorio exista y guardamos
                        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                        with open(file_path, "wb") as f:
                            f.write(message)
                        logger.info(f"[NODE] [FILE SAVED] {file_path}")
                        await websocket.send("ACK_FILE_SUCCESS")
                        result = file_path
                    else:
                        logger.warning("[NODE] Se recibieron bytes pero no se especificó file_path.")
                        await websocket.send("ACK_FILE_SUCCESS")
                        result = message  # Retorna los bytes crudos si no hay ruta
                else:
                    logger.info(f"[NODE] Mensaje de texto recibido: {message}")
                    await websocket.send("ACK_MESSAGE_SUCCESS")
                    result = message

                net_metrics = metrics.end_monitoring()

        except Exception as e:
            logger.error(f"[NODE] Error procesando la conexión entrante: {e}")
        finally:
            # Ya sea con éxito o fallo, avisamos que podemos cerrar el servidor
            stop_event.set()

    # Levantamos el servicio local de escucha
    async with websockets.serve(handler, host, port):
        try:
            # Esperamos a que ocurra el evento de recepción o que se agote el delay
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            logger.warning(f"[NODE] Tiempo agotado ({delay}s): El servidor central nunca se conectó.")

    return result, net_metrics