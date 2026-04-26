from loguru import logger
import sys

logger.info("Starting test script...")

try:
    logger.info("Importing cv2...")
    import cv2
    logger.info("Imported cv2.")
    
    logger.info("Importing numpy...")
    import numpy as np
    logger.info("Imported numpy.")
    
    logger.info("Importing torch...")
    import torch
    logger.info("Imported torch.")
    
    logger.info("Importing anomalib.engine.Engine...")
    from anomalib.engine import Engine
    logger.info("Imported anomalib.engine.Engine.")
    
    logger.info("Importing core.anomalib_engine...")
    from core.anomalib_engine import AnomalibEngine
    logger.info("Imported core.anomalib_engine.")
    
except Exception as e:
    logger.error(f"Error: {e}")

logger.info("Test script complete.")
