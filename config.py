"""
Configuración centralizada para el sistema de aprendizaje federado.
"""

# ── Hiperparámetros de entrenamiento ───────────────────────────────────
ROUNDS = 3
EPOCHS = 5
LEARNING_RATE = 1e-3
IN_FEATURES = 20

# ── Rutas de archivos ──────────────────────────────────────────────────
NODES_JSON = "nodes.json"
METRICS_CSV = "metrics.csv"
RECEIVED_FILES_PATH = "received_files"
CENTRAL_PATH = "CENTRAL"
MODEL_PATH = "model.pt"
DATA_PATH = "data/Hospital.csv"

# ── Configuración de modelos neuronales ────────────────────────────────
SEED = 42
DROPOUT_PROB = 0.3
MLP_HIDDEN1 = 120
MLP_HIDDEN2 = 84
MLPLRG_HIDDEN1 = 256
MLPLRG_HIDDEN2 = 128
MLPLRG_HIDDEN3 = 64

# ── Configuración de datos ────────────────────────────────────────────
TEST_SIZE = 0.2
BATCH_SIZE = 32
LABEL_COLUMN = "label"
RANDOM_STATE = 42

# ── Configuración de comunicación ─────────────────────────────────────
LISTENER_DURATION = 10
RECEIVED_MODEL_FILENAME = "received_model.pt"
ACK_IDENTIFIED = "Identified"
ACK_REGISTERED = "Registered"
ACK_FILE_SUCCESS = "File received successfully"
ACK_MESSAGE_SUCCESS = "Message received"
