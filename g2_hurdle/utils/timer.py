
import time
from .logging import get_logger

class Timer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None
        self.logger = get_logger("Timer")

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info(f"[START] {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        self.logger.info(f"[END  ] {self.name} - {dt:.2f}s")
