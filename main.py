import asyncio
import csv
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from config import ROUNDS, EPOCHS, LEARNING_RATE, IN_FEATURES, MODEL_PATH, CENTRAL_PATH, LISTENER_DURATION, NODES_JSON, METRICS_CSV
from connections.client import send, send_identified, send_file_identified, server_file
from connections.server import listener_ips, listener_nodes, listener_server
from model.create_model import create_model, MLP
from model.fed_model import ModelTrainer, federated_average
from utils import save_nodes, append_metrics


async def central_main(server_addr: str):
    """
    Entrada: dirección propia del servidor central (host:port).
    Salida: ninguna (orquesta K rondas de entrenamiento federado).
    """
    print("[CENTRAL] Creando modelo inicial...")
    create_model(in_features=IN_FEATURES, path=MODEL_PATH)

    print("[CENTRAL] Esperando nodos...")
    ips = await listener_ips(server_addr, duration=10)
    print(f"[CENTRAL] Nodos registrados: {ips}")
    
    os.makedirs(CENTRAL_PATH, exist_ok=True)
    nodes = save_nodes(ips, f"{CENTRAL_PATH}/{NODES_JSON}")
    n = nodes["n_nodes"]

    # {addr -> "Nodo_X"} sin incluir la clave "n_nodes"
    nodes_by_addr = {addr: nid for nid, addr in nodes.items() if nid != "n_nodes"}

    for round_n in range(1, ROUNDS + 1):
        print(f"\n[CENTRAL] ══ Ronda {round_n}/{ROUNDS} ══")

        # 1. Distribuir modelo global
        print(f"[CENTRAL] Distribuyendo modelo a {n} nodo(s)...")
        await server_file(server_addr, MODEL_PATH, n_clients=n, delay=LISTENER_DURATION * 6)

        # 2. Recoger modelo (.pt) y métricas (JSON) — cierra al completar todos
        print(f"[CENTRAL] Esperando modelos y métricas de {n} nodo(s)...")
        round_dir = f"{CENTRAL_PATH}/round_{round_n}"
        os.makedirs(round_dir, exist_ok=True)
        results = await listener_nodes(
            server_addr,
            nodes=nodes_by_addr,
            delay=300,
            save_path=round_dir,
        )

        # 3. Separar modelos y métricas
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

        append_metrics(metrics_list, round_n, f"{CENTRAL_PATH}/{METRICS_CSV}")

        if not model_paths:
            print("[CENTRAL] No se recibieron modelos. Abortando.")
            break

        # 4. FedAvg y actualizar modelo global
        print(f"[CENTRAL] Promediando {len(model_paths)} modelo(s)...")
        avg_state = federated_average(model_paths)
        torch.save(avg_state, MODEL_PATH)
        print(f"[CENTRAL] Modelo global actualizado → {MODEL_PATH}")

    print("\n[CENTRAL] Entrenamiento federado completado.")


async def client_main(addr: str, server_addr: str, node_id: str):
    """
    Entrada: dirección propia del nodo (host:port), dirección del servidor central,
             identificador del nodo.
    Salida: ninguna (ejecuta K rondas: descarga modelo, entrena, envía modelo y métricas).
    """
    folder     = f"NODE_{node_id}"
    data_path  = f"Hospital_{node_id}.csv"
    model_path = os.path.join(folder, "model.pt")
    os.makedirs(folder, exist_ok=True)

    # Registro inicial
    await asyncio.sleep(1)
    await send(server_addr, addr)

    for round_n in range(1, ROUNDS + 1):
        print(f"\n[NODE {node_id}] ══ Ronda {round_n}/{ROUNDS} ══")

        # 1. Descargar modelo global
        await asyncio.sleep(1)
        print(f"[NODE {node_id}] Descargando modelo global...")
        received = await listener_server(server_addr, delay=60, save_path=folder)
        if received is None:
            print(f"[NODE {node_id}] No se recibió modelo. Abortando.")
            break

        # 2. Entrenar con datos locales
        print(f"[NODE {node_id}] Entrenando con {data_path} por {EPOCHS} épocas...")
        architecture = MLP(in_features=IN_FEATURES)
        trainer = ModelTrainer(model_path=received, model_architecture=architecture)
        train_loader, test_loader = trainer.load_csv(data_path, label_col="label")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(architecture.parameters(), lr=LEARNING_RATE)
        trainer.fit(train_loader, criterion, optimizer, epochs=EPOCHS)
        metrics = trainer.evaluate(test_loader)
        trainer.save(model_path)

        # 3. Enviar modelo y métricas identificados
        await asyncio.sleep(1)
        print(f"[NODE {node_id}] Enviando modelo entrenado...")
        await send_file_identified(addr, server_addr, model_path)

        await send_identified(addr, server_addr, json.dumps(metrics))
        print(f"[NODE {node_id}] Métricas enviadas: {metrics}")

    print(f"\n[NODE {node_id}] Entrenamiento federado completado.")


def main():
    if len(sys.argv) != 5:
        print(
            "python3 main.py "
            "<true|false> "
            "<addr> "
            "<server_addr> "
            "<id>"
        )
        return

    central     = sys.argv[1].lower() == "true"
    addr        = sys.argv[2]
    server_addr = sys.argv[3]
    node_id     = sys.argv[4]

    if central:
        asyncio.run(central_main(server_addr))
    else:
        asyncio.run(client_main(addr, server_addr, node_id))


if __name__ == "__main__":
    main()