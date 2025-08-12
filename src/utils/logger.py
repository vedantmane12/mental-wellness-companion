"""
Logging configuration for the Mental Wellness Companion
"""
import sys
from pathlib import Path
from loguru import logger
from config.settings import LOG_LEVEL, LOG_FORMAT, DATA_DIR

# Remove default handler
logger.remove()

# Add console handler with color
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    colorize=True,
    backtrace=True,
    diagnose=True
)

# Add file handler for all logs
logger.add(
    DATA_DIR / "app.log",
    format=LOG_FORMAT,
    level="DEBUG",
    rotation="10 MB",
    retention="1 week",
    compression="zip"
)

# Add file handler for errors only
logger.add(
    DATA_DIR / "errors.log",
    format=LOG_FORMAT,
    level="ERROR",
    rotation="10 MB",
    retention="1 month",
    compression="zip",
    backtrace=True,
    diagnose=True
)

# Add specific handler for training logs
logger.add(
    DATA_DIR / "training.log",
    format=LOG_FORMAT,
    level="INFO",
    filter=lambda record: "training" in record["extra"],
    rotation="100 MB",
    retention="1 week"
)

# Create logger instance for training
training_logger = logger.bind(training=True)

# Export configured loggers
__all__ = ["logger", "training_logger"]