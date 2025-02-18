"""Logging configuration for the application."""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from .. import config

def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with both file and console handlers.
    
    Args:
        name: The name of the logger, typically __name__ of the module
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger
    
    # Create formatters
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_exception(logger: logging.Logger, e: Exception, context: str = ""):
    """Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        e: Exception to log
        context: Additional context about where/why the error occurred
    """
    import traceback
    error_msg = f"{context} - {str(e)}" if context else str(e)
    logger.error(error_msg)
    logger.debug(f"Full traceback:\n{traceback.format_exc()}")

def log_memory_usage(logger: logging.Logger, context: str = ""):
    """Log current memory usage.
    
    Args:
        logger: Logger instance
        context: Context for the memory usage check
    """
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    msg = f"Memory usage{' (' + context + ')' if context else ''}: "
    msg += f"{memory_info.rss / 1024 / 1024:.1f} MB"
    logger.debug(msg)