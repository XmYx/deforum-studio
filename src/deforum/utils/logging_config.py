import os
import sys
import logging
from loguru import logger

from .constants import root_path

def setup_file_logging(log_file, log_level, log_max_bytes, log_backup_count):
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} - {file}:{line} - {level} - {message}",
        rotation=log_max_bytes,
        retention=log_backup_count,
        backtrace=True,
        diagnose=True,
    )

def setup_console_logging(log_level):
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <cyan>{file}:{line}</cyan> - <lvl>{level}</lvl> - <lvl>{message}</lvl>",
        backtrace=True,
        diagnose=True,
    )

def setup_logging(config=None):
    log_level = os.environ.get('DEFORUM_LOG_LEVEL', 'DEBUG')
    log_dir = os.path.join(root_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.environ.get('DEFORUM_LOG_FILE', os.path.join(root_path, 'logs/app.log'))
    log_max_bytes = int(os.environ.get('DEFORUM_LOG_MAX_BYTES', 10485760))
    log_backup_count = int(os.environ.get('DEFORUM_LOG_BACKUP_COUNT', 10))

    logger.remove()  # Remove the default logger

    setup_file_logging(log_file, log_level, log_max_bytes, log_backup_count)
    setup_console_logging(log_level)

    return logger

class LoguruHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_root_logger():
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    root_logger.addHandler(LoguruHandler())

setup_root_logger()
logger = setup_logging()