#!/usr/bin/env python3
"""
Script de prueba para verificar que el sistema de logging funciona correctamente.
"""

from logging_config import get_logger
import time

logger = get_logger(__name__)

def test_logging():
    """Prueba todos los niveles de logging."""
    
    logger.info("=" * 50)
    logger.info("INICIANDO PRUEBA DE LOGGING")
    logger.info("=" * 50)
    
    # Prueba INFO
    logger.info("[TEST] Mensaje de información (INFO)")
    time.sleep(0.5)
    
    # Prueba WARNING
    logger.warning("[TEST] Mensaje de advertencia (WARNING)")
    time.sleep(0.5)
    
    # Prueba ERROR
    logger.error("[TEST] Mensaje de error (ERROR)")
    time.sleep(0.5)
    
    # Prueba CRITICAL
    logger.critical("[TEST] Mensaje crítico (CRITICAL)")
    time.sleep(0.5)
    
    # Prueba con variables
    test_var = "ejemplo_de_variable"
    node_addr = "192.168.1.100:8765"
    logger.info(f"[TEST] Variable 1: {test_var}, Variable 2: {node_addr}")
    
    logger.info("=" * 50)
    logger.info("PRUEBA COMPLETADA - Verifica los timestamps en los logs")
    logger.info(f"Los logs se guardan en: ./logs/")
    logger.info("=" * 50)

if __name__ == "__main__":
    test_logging()
