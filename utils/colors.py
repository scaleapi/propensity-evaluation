import logging
import colorama
from colorama import Fore, Style

colorama.init()

class BaseColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages based on their level"""

    COLORS = {
        logging.INFO: Fore.WHITE,
        logging.DEBUG: Fore.BLUE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA
    }

    def format(self, record):
        # Add color to the log level name and message
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)
