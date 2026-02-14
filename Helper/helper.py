import logging
from langchain_community.document_loaders import PyPDFLoader
import os

class Logger:
    def __init__(self, name="DocQueryAI", log_file="logs/app.log"):
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # allow all levels

        # Prevent duplicate handlers
        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def debug(self, message):
        self.logger.debug(message)



def load_pdf(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents