"""
Configuración centralizada de logging para el sistema de aprendizaje federado.
Incluye timestamps, niveles de severidad y salida a consola y archivo.
"""

import logging
import logging.handlers
import os
from datetime import datetime

# Crear directorio de logs si no existe
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Nombre del archivo de log con timestamp
log_filename = os.path.join(LOG_DIR, f"federated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Formato con timestamps detallados
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Crear logger raíz
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Eliminar handlers existentes para evitar duplicados
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Handler para consola (nivel INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Handler para archivo (nivel DEBUG)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger con el nombre especificado.
    Entrada: nombre del módulo (típicamente __name__).
    Salida: instancia de logging.Logger configurada.
    """
    return logging.getLogger(name)
