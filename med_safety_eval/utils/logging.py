import logging
import os

def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger for the given name."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Default level to INFO, can be overridden by env var
        level_name = os.getenv("MED_SAFETY_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
