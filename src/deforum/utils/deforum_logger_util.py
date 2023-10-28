import os
import textwrap
import time
from datetime import datetime


class Logger:
    """
    Logger class for logging messages to a file with optional timestamps.

    Provides functionalities to start a logging session, log messages, and close the logging session.
    """

    def __init__(self, root_path: str):
        """
        Initialize the Logger object.

        Args:
            root_path (str): Root directory path where the log files will be stored.
        """
        self.root_path = root_path
        self.log_file = None
        self.current_datetime = datetime.now()
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.terminal_width = self.get_terminal_width()
        self.start_time = time.time()
        self.log_entries = []

    @staticmethod
    def get_terminal_width() -> int:
        """Get the width of the terminal.

        Returns:
            int: Width of the terminal, or a default width if the terminal size cannot be determined.
        """
        try:
            import shutil
            return shutil.get_terminal_size().columns
        except (ImportError, AttributeError):
            # Default width
            return 80

    def start_session(self):
        """Start a logging session by creating or appending to a log file."""
        year, month, day = self.current_datetime.strftime('%Y'), self.current_datetime.strftime(
            '%m'), self.current_datetime.strftime('%d')
        log_path = os.path.join(self.root_path, 'logs', year, month, day)
        os.makedirs(log_path, exist_ok=True)

        self.log_file = open(
            os.path.join(log_path, f"metrics_{self.timestamp.replace(' ', '_').replace(':', '_')}.log"), "a")
        self.log_file.write("=" * self.terminal_width + "\n")
        self.log_file.write("Log Session Started: " + self.timestamp.center(self.terminal_width - 20) + "\n")
        self.log_file.write("=" * self.terminal_width + "\n")
        self.log_file.close()

    def log(self, message: str, timestamped: bool = True):
        """
        Log a message to the log file.

        Args:
            message (str): The message to be logged.
            timestamped (bool, optional): If True, add a timestamp prefix to the message. Default is True.
        """

        time_now = time.time()
        duration = (time_now - self.start_time) * 1000
        self.start_time = time_now
        if timestamped:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = f"[{timestamp} {duration:.2f} ms] {message}"

        # Wrap the message to the terminal width
        wrapped_text = "\n".join(textwrap.wrap(message, width=self.terminal_width))
        self.log_entries.append(wrapped_text)
    def dump(self):
        """Dump the internal log entries to the log file and clear the internal list."""
        if self.log_file:
            with open(self.log_file.name, "a") as log:
                for entry in self.log_entries:
                    log.write(entry + "\n")
            self.log_entries.clear()

    def print_logs(self):
        """Print the current logs stored in the internal list."""
        for entry in self.log_entries:
            print(entry)

    def __call__(self, message: str, timestamped: bool = True, *args, **kwargs):
        """Allow the Logger object to be called as a function."""
        self.log(message, timestamped)

    def close_session(self):
        """End the logging session."""
        if self.log_file:
            with open(self.log_file.name, "a") as log:
                log.write("\n" + "=" * self.terminal_width + "\n")
                log.write("Log Session Ended".center(self.terminal_width) + "\n")
                log.write("=" * self.terminal_width + "\n")
