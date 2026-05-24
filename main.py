import asyncio
import os
import sys
import time
from connections.client import send, send_file_to_nodes
from connections.server import listener_ips, listener_server
from config import LISTENER_DURATION, MODEL_PATH, IN_FEATURES
from utils import get_ipport
from federated import main as fed
from logging_config import get_logger
from model.create_model import create_model

logger = get_logger(__name__)

async def sharing(ip_father: str, ip: str, ips_children: list[str]):
    
    ips = []
    _, listen_port = get_ipport(ip)
    # Usamos 0.0.0.0 para que escuche en todas las tarjetas de red (Docker o Raspberrys)
    local_listen_addr = f"0.0.0.0:{listen_port}"

    if ip_father == ip:
        logger.info("[CENTRAL HIER] Creando modelo")
        create_model(in_features=IN_FEATURES, path=MODEL_PATH)
        logger.info("[CENTRAL HIER] Modelo creado")

    if ip_father != ip:
        # Primero registrarse con el padre
        logger.info("[HIER] Enviando IP propia al padre...")
        await send(ip_father, ip)

        logger.info("[HIER] Escuchando IPs hijas...")
        ips = await listener_ips(local_listen_addr, LISTENER_DURATION)
        logger.info(f"[HIER] Hijos registrados: {ips}")

        # Esperar el modelo del padre
        await get_model(local_listen_addr, LISTENER_DURATION * 10)

    else:
        # Raíz: escucha hijos y distribuye
        logger.info("[HIER] Soy raíz. Escuchando IPs hijas...")
        ips = await listener_ips(local_listen_addr, LISTENER_DURATION * 2)
        logger.info(f"[HIER] Hijos registrados: {ips}")

    if len(ips):
        await send_model(ips)
    else:
        logger.info("[HIER] Soy nodo hoja, sin hijos. Distribución completada.")

    for ip in ips:
        ips_children.append(ip) 





async def get_model(local_listen_addr: str, delay: int = LISTENER_DURATION):
    logger.info("[HIER] Esperando modelo del padre...")

    received = await listener_server(local_listen_addr, delay, file_path=MODEL_PATH)
    if received is None:
        logger.error("[HIER] No se recibió modelo. Abortando.")
        return
    
async def send_model(ips_children: list[str]):
    logger.info(f"[HIER] Distribuyendo modelo a {len(ips_children)} hijo(s)...")
    await send_file_to_nodes(ips_children, MODEL_PATH, delay=LISTENER_DURATION * 6)
    #Se elimina el modelo de todos los nodos intermedios
    os.remove(MODEL_PATH)
    
async def distribute_model(ips_children: list[str], ip_father: str, ip: str):
    _, listen_port = get_ipport(ip)
    # Usamos 0.0.0.0 para que escuche en todas las tarjetas de red (Docker o Raspberrys)
    local_listen_addr = f"0.0.0.0:{listen_port}"

    if len(ips_children) and (ip_father == ip):
        await send_model(ips_children)

    elif len(ips_children):
        await get_model(local_listen_addr, LISTENER_DURATION * 100)
        await send_model(ips_children)

    else:
        await get_model(local_listen_addr, LISTENER_DURATION * 100)
        logger.info("[HIER] Soy nodo hoja, sin hijos. Distribución completada.")



def main():
    ip = sys.argv[1]          # dirección propia host:port
    ip_father = sys.argv[2]   # "null" si es raíz

    ips = []
    asyncio.run(sharing(ip_father, ip, ips))

    leaf = len(ips) == 0

    N_ROUNDS = 5

    for round in range(N_ROUNDS):
        logger.info(f"[HIER] Ronda jerárquica número {round}")

        if leaf:
            fed(central=False, addr=ip, server_addr=ip_father)
        elif (not leaf) and (ip_father == ip):
            fed(central=True, addr=ip, server_addr=ip, childs=ips)
        else:
            fed(central=True, addr=ip, server_addr=ip_father, childs=ips)
            fed(central=False, addr=ip, server_addr=ip_father)
        
        if round < (N_ROUNDS - 1):
            asyncio.run(distribute_model(ips, ip_father, ip))
        
        logger.info(f"[HIER] Ronda jerárquica número {round} terminada")

    logger.info(f"[HIER] Proceso jerárquico terminado")

if __name__ == "__main__":
    main()