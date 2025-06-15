"""
Logging configuration and utilities.
"""

import logging
from typing import Optional


def setup_logging(level: str = "INFO", logger_name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        logger_name: Name of the logger (defaults to package name)
    
    Returns:
        Configured logger instance
    """
    logger_name = logger_name or __name__.split('.')[0]
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper()))
    return logger
