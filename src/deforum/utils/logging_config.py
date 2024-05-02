import logging
import logging.handlers
import os

from .constants import root_path
# Retrieve the home directory using the HOME environment variable
home_dir = os.getenv('HOME')

# Define the path for the 'deforum' directory within the home directory
#root_path = os.path.join(home_dir, 'deforum')

def setup_logging(config=None):
    log_level = os.environ.get('DEFORUM_LOG_LEVEL', 'DEBUG')
    log_file = os.environ.get('DEFORUM_LOG_FILE', os.path.join(root_path,'logs/app.log'))
    log_max_bytes = int(os.environ.get('DEFORUM_LOG_MAX_BYTES', 10485760))
    log_backup_count = int(os.environ.get('DEFORUM_LOG_BACKUP_COUNT', 10))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger('deforum')
    logger.setLevel(log_level)

    # Optionally add file handler:
    if log_file is not None:
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=log_max_bytes, backupCount=log_backup_count)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging()