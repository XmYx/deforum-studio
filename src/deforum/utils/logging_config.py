import logging
import logging.handlers
import os
from deforum.utils.constants import LogConfig, config

def setup_logging(conf : LogConfig  = config):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('deforum')
    logger.setLevel(conf.log_level)

    # Optionally add file handler:
    if conf.log_to_file and conf.log_file is not None:
        os.makedirs(os.path.dirname(conf.log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(conf.log_file, maxBytes=conf.log_max_bytes, backupCount=conf.log_backup_count)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug(f"Running Deforum with config: {config}")

    return logger


logger = setup_logging()