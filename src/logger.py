"""Functions to log info"""
# =================
# ==== IMPORTS ====
# =================

import logging


# ===================
# ==== FUNCTIONS ====
# ===================

def get_console_logger(name: str='test') -> logging.Logger:
    
    # Create logger if it doesn't exist
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add console handler to the logger
        logger.addHandler(console_handler)

    return logger