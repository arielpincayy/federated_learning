import asyncio
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from config import ROUNDS, EPOCHS, LEARNING_RATE, IN_FEATURES, LISTENER_DURATION, NODES_JSON, METRICS_CSV, MODEL_PATH, DATA_PATH, SLEEP_INTERVAL, POST_ROUND_DELAY, NODES_LISTENER_DELAY, LABEL_COLUMN
from connections.client import send_identified, send_file_identified, send_file_to_nodes, send_message_to_nodes, send
from connections.server import listener_nodes, listener_server
from model.create_model import MLP
from model.fed_model import ModelTrainer, federated_average
from utils import save_nodes, append_metrics, get_ipport
from logging_config import get_logger

logger = get_logger(__name__)

async def central_main(addr: str, ips: list[str]):
    """
    Entrada: dirección propia del servidor central (host:port).
    """

    nodes = save_nodes(ips, NODES_JSON)
    n = nodes["n_nodes"]

    nodes_by_addr = {a: nid for nid, a in nodes.items() if nid != "n_nodes"}

    #print("[CENTRAL] Enviando mensaje START")
    #await send_message_to_nodes(node_addrs=ips, message="start", delay=LISTENER_DURATION)
    #print("[CENTRAL] Mensaje START enviado")

    # Extraemos el puerto en el que este nodo específico debe escuchar
    _, listen_port = get_ipport(addr)
    # Usamos 0.0.0.0 para que escuche en todas las tarjetas de red (Docker o Raspberrys)
    local_listen_addr = f"0.0.0.0:{listen_port}"

    for round_n in range(1, ROUNDS + 1):
        logger.info(f"\n[CENTRAL] ══ Ronda {round_n}/{ROUNDS} ══")

        if round_n > 1:
            await asyncio.sleep(POST_ROUND_DELAY)
            logger.info(f"[CENTRAL] Distribuyendo modelo agregado a {n} nodo(s)...")
            await send_file_to_nodes(ips, MODEL_PATH, delay=LISTENER_DURATION * 6)

        logger.info(f"[CENTRAL] Esperando modelos y métricas de {n} nodo(s)...")
        round_dir = f"round_{round_n}"
        os.makedirs(round_dir, exist_ok=True)
        results = await listener_nodes(
            local_listen_addr,
            nodes=nodes_by_addr,
            delay=NODES_LISTENER_DELAY * 100,
            save_path=round_dir,
        )

        metrics_list = []
        model_paths  = []
        for node_id, received in results.items():
            for item in received:
                if isinstance(item, str) and os.path.isfile(item) and item.endswith(".pt"):
                    model_paths.append(item)
                elif isinstance(item, str):
                    try:
                        m = json.loads(item)
                        m["node"] = node_id
                        metrics_list.append(m)
                    except json.JSONDecodeError:
                        pass

        append_metrics(metrics_list, round_n, METRICS_CSV)

        if not model_paths:
            logger.error("[CENTRAL] No se recibieron modelos. Abortando.")
            break

        logger.info(f"[CENTRAL] Promediando {len(model_paths)} modelo(s)...")
        avg_state = federated_average(model_paths)
        torch.save(avg_state, MODEL_PATH)
        logger.info(f"[CENTRAL] Modelo global actualizado → {MODEL_PATH}")

    logger.info("\n[CENTRAL] Entrenamiento federado completado.")


async def client_main(addr: str, server_addr: str):
    """
    Entrada: dirección propia del nodo (host:port), dirección del servidor central.
    """
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    # === CAMBIO CLAVE AQUÍ ===
    # Extraemos el puerto en el que este nodo específico debe escuchar
    _, listen_port = get_ipport(addr)
    # Usamos 0.0.0.0 para que escuche en todas las tarjetas de red (Docker o Raspberrys)
    local_listen_addr = f"0.0.0.0:{listen_port}"

    #print(f"[NODE] Esperando mensaje START en el puerto {listen_port}...")
    #received = await listener_server(local_listen_addr, delay=LISTENER_DURATION * 100)
    #if received is None:
    #    print(f"[NODE] No se recibió mensaje START. Abortando.")
    #    return

    for round_n in range(1, ROUNDS + 1):
        logger.info(f"\n[NODE] ══ Ronda {round_n}/{ROUNDS} ══")

        if round_n > 1:
            logger.info(f"[NODE] Descargando modelo agregado desde el servidor central...")
            # === CAMBIO CLAVE AQUÍ ===
            # El nodo se queda esperando pasivamente a que el central le empuje el archivo .pt
            received = await listener_server(local_listen_addr, delay=LISTENER_DURATION * 6, file_path=MODEL_PATH)
            if received is None:
                logger.error(f"[NODE] No se recibió modelo. Abortando.")
                break

        logger.info(f"[NODE] Entrenando con {DATA_PATH} por {EPOCHS} épocas...")
        architecture = MLP(in_features=IN_FEATURES)
        trainer = ModelTrainer(model_path=MODEL_PATH, model_architecture=architecture)
        train_loader, test_loader = trainer.load_csv(DATA_PATH, label_col=LABEL_COLUMN)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(architecture.parameters(), lr=LEARNING_RATE)
        trainer.fit(train_loader, criterion, optimizer, epochs=EPOCHS)
        metrics = trainer.evaluate(test_loader)
        trainer.save(MODEL_PATH)

        await asyncio.sleep(SLEEP_INTERVAL)
        logger.info(f"[NODE] Enviando modelo entrenado...")
        await send_file_identified(addr, server_addr, MODEL_PATH)
        await send_identified(addr, server_addr, json.dumps(metrics))
        logger.info(f"[NODE] Métricas enviadas: {metrics}")

    logger.info(f"\n[NODE] Entrenamiento federado completado.")


def main(central: bool, addr: list, server_addr: str, childs: list[str]=None):
    #if len(sys.argv) != 4:
    #    print(
    #        "Uso: python3 main.py <true|false> <addr> <server_addr>\n"
    #        "  true  → servidor central\n"
    #        "  false → nodo cliente"
    #    )
    #    return
    
    #central     = sys.argv[1].lower() == "true"
    #addr        = sys.argv[2]
    #server_addr = sys.argv[3]
    #if central:
    #    childs      = sys.argv[4].split(',')

    if central:
        asyncio.run(central_main(addr, childs))
    else:
        asyncio.run(client_main(addr, server_addr))


if __name__ == "__main__":
    main()