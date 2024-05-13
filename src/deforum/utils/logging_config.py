import os
import sys
import logging
from loguru import logger

from .constants import config

def setup_file_logging(log_file, log_level, log_max_bytes, log_backup_count):
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} - {file}:{line} - {level} - {message}",
        rotation=log_max_bytes,
        retention=log_backup_count,
        backtrace=True,
        diagnose=True,
        enqueue=True
    )

def setup_console_logging(log_level):
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <cyan>{file}:{line}</cyan> - <lvl>{level}</lvl> - <lvl>{message}</lvl>",
        backtrace=True,
        diagnose=True,
        enqueue=True
    )

def setup_logging():
    if (config.log_to_file and config.log_dir is not None):
        os.makedirs(config.log_dir, exist_ok=True)
    log_file = os.path.join(config.log_dir, 'app.log')

    logger.remove()  # Remove the default logger

    setup_file_logging(log_file, config.log_level, config.log_max_bytes, config.log_backup_count)
    setup_console_logging(config.log_level)

    logger.debug(f"Running Deforum with config: {config}")

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


# import logging
# import logging.handlers
# import os
#
# from .constants import config
# # Retrieve the home directory using the HOME environment variable
# home_dir = os.getenv('HOME')
#
# # Define the path for the 'deforum' directory within the home directory
# #root_path = os.path.join(home_dir, 'deforum')
#
# def setup_logging(config=None):
#     log_level = os.environ.get('DEFORUM_LOG_LEVEL', 'DEBUG')
#     log_dir = os.path.join(config.root_path,'logs')
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.environ.get('DEFORUM_LOG_FILE', os.path.join(config.root_path,'logs/app.log'))
#     log_max_bytes = int(os.environ.get('DEFORUM_LOG_MAX_BYTES', 10485760))
#     log_backup_count = int(os.environ.get('DEFORUM_LOG_BACKUP_COUNT', 10))
#
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
#     logger = logging.getLogger('deforum')
#     logger.setLevel(log_level)
#
#     # Optionally add file handler:
#     if log_file is not None:
#         file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=log_max_bytes, backupCount=log_backup_count)
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)
#
#     return logger
#
#
# logger = setup_logging(config)