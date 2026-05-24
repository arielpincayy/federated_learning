# Federated Learning Hierarchical System

A hierarchical federated learning project in Python, designed to train an MLP model on tabular data distributed across multiple nodes.

## Description

This project implements a two-level federated architecture:

- Central node (root) that coordinates global aggregation.
- Intermediate and leaf nodes that receive the model, train locally, and send updates.

The system uses WebSockets for node-to-node communication and PyTorch for model training.

## Project Structure

- `main.py`: main entry point for running the hierarchical federated learning flow.
- `federated.py`: central server and client logic for classic federated training.
- `config.py`: global configuration, hyperparameters, and file paths.
- `create_model.py`: generates an initial `model.pt` if needed.
- `logging_config.py`: logger configuration for console and file output.
- `utils.py`: IP/port utilities, node saving, and metrics logging.
- `connections/client.py`: functions for sending messages and files over WebSockets.
- `connections/server.py`: WebSocket servers for receiving nodes, models, and metrics.
- `model/create_model.py`: MLP architecture definition and initial model creation.
- `model/fed_model.py`: local training classes and federated averaging (FedAvg).
- `data/`: sample hospital dataset files.
- `Dockerfile` and `docker-compose.yml`: container deployment.

## Requirements

- Python 3.12
- `torch`
- `websockets`
- `pandas`
- `scikit-learn`

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local Usage

### Run the central (root) node

```bash
python main.py 0.0.0.0:8765 0.0.0.0:8765
```

### Run a child node

```bash
python main.py node1:8766 0.0.0.0:8765
```

### Run a leaf node connected to an intermediate node

```bash
python main.py node3:8768 node1:8766
```

> Note: `main.py` uses `ip_father == ip` to identify the root node. Child nodes send their address to the parent, receive the model, and then train locally.

## Docker Usage

Build the image:

```bash
docker build -t federared-system_image:latest .
```

Start the architecture defined in `docker-compose.yml`:

```bash
docker compose up --build
```

The `docker-compose.yml` file defines a hierarchical tree with a central node and several child nodes connected to different parents.

## Training Flow

1. The root creates or loads an initial model (`model.pt`).
2. The root and intermediate nodes register and share their child addresses.
3. The model is distributed hierarchically to all leaf nodes.
4. Each leaf node trains locally on its CSV dataset.
5. Nodes send their trained models and metrics back to the central server.
6. The central server computes a weighted average of model weights (`FedAvg`) and updates the global model.
7. The loop repeats for the number of rounds configured in `config.py`.

## Generated Files

- `model.pt`: global model saved after aggregation.
- `metrics.csv`: metrics from each node per round.
- `nodes.json`: node list registered by the central server.
- `logs/`: execution logs.

## Configuration

Main values are defined in `config.py`:

- `ROUNDS`: federated learning rounds.
- `EPOCHS`: local training epochs.
- `LEARNING_RATE`: learning rate.
- `IN_FEATURES`: number of input features.
- `LISTENER_DURATION`: connection listening duration.
- `DATA_PATH`: CSV path used by each node.

## Important Notes

- For Docker execution, each node mounts a distinct CSV file as `data/Hospital.csv`.
- The project expects tabular data with the label column `is_premature_ncd`.
- Nodes use `BCEWithLogitsLoss` for binary classification.

## Possible Extensions

- Add data validation before training.
- Include a dynamic node orchestrator.
- Adapt the hierarchical topology to more layers and subtree sizes.
- Add encryption or differential privacy.

---

### Contact

Project created for distributed federated learning experiments with PyTorch and WebSockets.
