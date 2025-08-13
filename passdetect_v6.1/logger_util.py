#!/usr/bin/env python3
import logging
import os
from logging.handlers import RotatingFileHandler
from multiprocessing import Queue, current_process
from queue import Empty
import threading
import time

def _listener_process(queue: Queue, logfile: str):
    logger = logging.getLogger("passdetect")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fmt = logging.Formatter(fmt="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    while True:
        try:
            rec = queue.get(timeout=0.5)
        except Empty:
            continue
        if rec is None:
            break
        logger.handle(rec)

def start_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, "run.log")
    queue = Queue()
    listener = threading.Thread(target=_listener_process, args=(queue, logfile), daemon=True)
    listener.start()
    return queue, listener

def get_worker_logger(queue: Queue, name="passdetect.worker"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    class QueueHandler(logging.Handler):
        def __init__(self, q): super().__init__(); self.q = q
        def emit(self, record): self.q.put(record)
    logger.handlers = []
    logger.addHandler(QueueHandler(queue))
    return logger

def stop_logger(queue: Queue, listener_thread: threading.Thread):
    try:
        queue.put(None)
    except Exception:
        pass
    # give it a moment to flush
    time.sleep(0.5)
