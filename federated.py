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

    # Extraemos el puerto en el que este nodo específico debe escuchar
    _, listen_port = get_ipport(addr)
    # Usamos 0.0.0.0 para que escuche en todas las tarjetas de red (Docker o Raspberrys)
    local_listen_addr = f"0.0.0.0:{listen_port}"

    for round_n in range(1, ROUNDS + 1):
        logger.info(f"\n[CENTRAL] ══ Ronda {round_n}/{ROUNDS} ══")

        if round_n > 1:
            await asyncio.sleep(POST_ROUND_DELAY)
            logger.info(f"[CENTRAL] Distribuyendo modelo agregado a {n} nodo(s)...")
            node_dist_metrics = await send_file_to_nodes(ips, MODEL_PATH, delay=LISTENER_DURATION * 6)
            logger.info(f"[CENTRAL] Métricas de distribución recopiladas de {len(node_dist_metrics)} nodo(s)")

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
                elif isinstance(item, dict):
                    # Las métricas de red vienen como dict
                    logger.info(f"[CENTRAL] Métricas de red capturadas para {node_id}: {item}")
                elif isinstance(item, str):
                    try:
                        m = json.loads(item)
                        m["node"] = node_id
                        metrics_list.append(m)
                    except json.JSONDecodeError:
                        pass

        converged = append_metrics(metrics_list, round_n, path=METRICS_CSV, K=3)

        if not model_paths:
            logger.error("[CENTRAL] No se recibieron modelos. Abortando.")
            break

        logger.info(f"[CENTRAL] Promediando {len(model_paths)} modelo(s)...")
        avg_state = federated_average(model_paths)
        torch.save(avg_state, MODEL_PATH)
        logger.info(f"[CENTRAL] Modelo global actualizado → {MODEL_PATH}")
        
        if round_n < ROUNDS:
            logger.info(f"[CENTRAL] Enviando mensaje de convergencia: CONVERGED {converged}")
            if converged:
               await send_message_to_nodes(ips, "CONVERGED", LISTENER_DURATION)
               break
            else:
                await send_message_to_nodes(ips, "NOT CONVERGED", LISTENER_DURATION)
            logger.info("[CENTRAL] Mensaje de convergencia enviado")

    logger.info("\n[CENTRAL] Entrenamiento federado completado.")


async def client_main(addr: str, server_addr: str):
    """
    Entrada: dirección propia del nodo (host:port), dirección del servidor central.
    """
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    _, listen_port = get_ipport(addr)
    # Usamos 0.0.0.0 para que escuche en todas las tarjetas de red (Docker o Raspberrys)
    local_listen_addr = f"0.0.0.0:{listen_port}"

    for round_n in range(1, ROUNDS + 1):
        logger.info(f"\n[NODE] ══ Ronda {round_n}/{ROUNDS} ══")

        net_metrics_round = {}  # Almacenar métricas de red de esta ronda

        if round_n > 1:
            logger.info(f"[NODE] Descargando modelo agregado desde el servidor central...")
            received, net_metrics_rx = await listener_server(local_listen_addr, delay=LISTENER_DURATION * 6, file_path=MODEL_PATH)
            if received is None:
                logger.error(f"[NODE] No se recibió modelo. Abortando.")
                break
            net_metrics_round.update(net_metrics_rx)
            logger.info(f"[NODE] Métricas de red RX: BW={net_metrics_rx.get('net_bandwidth_rx_kbps', 0):.2f}kbps")

        logger.info(f"[NODE] Entrenando con {DATA_PATH} por {EPOCHS} épocas...")
        architecture = MLP(in_features=IN_FEATURES)
        trainer = ModelTrainer(model_path=MODEL_PATH, model_architecture=architecture)
        train_loader, test_loader = trainer.load_csv(DATA_PATH, label_col=LABEL_COLUMN)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(architecture.parameters(), lr=LEARNING_RATE)
        trainer_time, trainer_loss = trainer.fit(train_loader, criterion, optimizer, epochs=EPOCHS)
        metrics = trainer.evaluate(test_loader)
        metrics['trainning_time'] = trainer_time
        metrics['loss'] = trainer_loss
        trainer.save(MODEL_PATH)

        await asyncio.sleep(SLEEP_INTERVAL)
        logger.info(f"[NODE] Enviando modelo entrenado...")
        success, net_metrics_tx = await send_file_identified(addr, server_addr, MODEL_PATH)
        net_metrics_round.update(net_metrics_tx)
        if success:
            logger.info(f"[NODE] Métricas de red TX: BW={net_metrics_tx.get('net_bandwidth_tx_kbps', 0):.2f}kbps")
        
        # Integrar métricas de red con métricas de entrenamiento
        metrics.update(net_metrics_round)
        
        await send_identified(addr, server_addr, json.dumps(metrics))
        logger.info(f"[NODE] Métricas enviadas: {metrics}")
        
        if round_n < ROUNDS:
            logger.info("[CLIENT] Esperando posible convergencia")
            received, _ = await listener_server(local_listen_addr, LISTENER_DURATION * 100)
            converged = (received == "CONVERGED")
            logger.info(f"[CLIENT] Convergencia: {converged}")
            if converged:
                logger.info("[NODE] Entrenamiento centralizado ha convergido")
                break

    logger.info("\n[NODE] Entrenamiento federado completado.")


def main(central: bool, addr: list, server_addr: str, childs: list[str]=None):

    if central:
        asyncio.run(central_main(addr, childs))
    else:
        asyncio.run(client_main(addr, server_addr))


if __name__ == "__main__":
    main()