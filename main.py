import asyncio
import os
import sys
import time
from connections.client import send, send_file_to_nodes
from connections.server import listener_ips, listener_server
from config import LISTENER_DURATION, MODEL_PATH
from utils import get_ipport
from federated import main as fed

async def sharing(ip_father: str, ip: str, ips_children: list[str]):
    
    ips = []
    if ip_father != ip:
        # Primero registrarse con el padre
        print("[HIER] Enviando IP propia al padre...")
        await send(ip_father, ip)

        print("[HIER] Escuchando IPs hijas...")
        ips = await listener_ips(ip, LISTENER_DURATION)
        print(f"[HIER] Hijos registrados: {ips}")

        # Esperar el modelo del padre
        print("[HIER] Esperando modelo del padre...")
        _, listen_port = get_ipport(ip)
        # Usamos 0.0.0.0 para que escuche en todas las tarjetas de red (Docker o Raspberrys)
        local_listen_addr = f"0.0.0.0:{listen_port}"

        received = await listener_server(ip, LISTENER_DURATION * 100, file_path=MODEL_PATH)
        if received is None:
            print("[HIER] No se recibió modelo. Abortando.")
            return

    else:
        # Raíz: escucha hijos y distribuye
        print("[HIER] Soy raíz. Escuchando IPs hijas...")
        ips = await listener_ips(ip, LISTENER_DURATION * 2)
        print(f"[HIER] Hijos registrados: {ips}")

    if ips:
        print(f"[HIER] Distribuyendo modelo a {len(ips)} hijo(s)...")
        time.sleep(2)
        await send_file_to_nodes(ips, MODEL_PATH, delay=LISTENER_DURATION * 6)
        # Se elimina el modelo de todos los nodos intermedios
        os.remove(MODEL_PATH)
    else:
        print("[HIER] Soy nodo hoja, sin hijos. Distribución completada.")

    for ip in ips:
        ips_children.append(ip)


def main():
    ip = sys.argv[1]          # dirección propia host:port
    ip_father = sys.argv[2]   # "null" si es raíz

    ips = []
    asyncio.run(sharing(ip_father, ip, ips))

    leaf = len(ips) == 0

    if leaf:
        fed(central=False, addr=ip, server_addr=ip_father)
    elif (not leaf) and (ip_father == ip):
        fed(central=True, addr=ip, server_addr=ip, childs=ips)
    else:
        fed(central=True, addr=ip, server_addr=ip_father, childs=ips)
        fed(central=False, addr=ip, server_addr=ip_father)
        


if __name__ == "__main__":
    main()