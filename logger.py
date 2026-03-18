import functools
import logging
import os
import time
import json
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler

LOG_DIR = "./logs"
SYSTEM_LOG_FILE = os.path.join(LOG_DIR, "system.log")
ACCESS_LOG_FILE = os.path.join(LOG_DIR, "access.json")

os.makedirs(LOG_DIR, exist_ok=True)


class ISTFormatter(logging.Formatter):
    """Custom formatter to use Indian Standard Time (IST) for logs."""

    def formatTime(self, record, datefmt=None):
        ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        return ist_time.strftime(datefmt if datefmt else "%Y-%m-%d %H:%M:%S")


class JSONFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record):
        log_entry = record.msg if isinstance(record.msg, dict) else {"message": record.getMessage()}
        ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        log_entry["timestamp"] = ist_time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry["level"] = record.levelname
        return json.dumps(log_entry)


def get_logger(name="system"):
    """Returns a logger for system/debug logs."""
    logger = logging.getLogger(f"system_{name}")

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = ISTFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = RotatingFileHandler(SYSTEM_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_formatter = ISTFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_access_logger():
    """Returns a logger for structured access logs."""
    logger = logging.getLogger("access")

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        file_handler = RotatingFileHandler(ACCESS_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


SYSTEM_LOGGER = get_logger("Final_API")
ACCESS_LOGGER = get_access_logger()


def log_request(endpoint_type):
    """Decorator to log API requests and responses in JSON format."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from flask import request, g
            request_id = str(os.urandom(4).hex())
            g.request_id = request_id
            # For endpoints like /transcriptions/formatted that use 'conversation_id'
            # For others we might get it from form data or generate one
            conv_id = request.form.get("conversation_id", "unknown")
            
            start_time = time.time()
            SYSTEM_LOGGER.info(f"[{request_id}] START {request.path} | Conv: {conv_id}")
            
            try:
                response = func(*args, **kwargs)
                duration = time.time() - start_time
                status_code = response[1] if isinstance(response, tuple) else 200
                
                ACCESS_LOGGER.info({
                    "request_id": request_id,
                    "conversation_id": conv_id,
                    "endpoint": request.path,
                    "type": endpoint_type,
                    "status": "success" if status_code < 400 else "failed",
                    "status_code": status_code,
                    "duration": round(duration, 3)
                })
                SYSTEM_LOGGER.info(f"[{request_id}] END {request.path} | Duration: {duration:.3f}s | Status: {status_code}")
                return response
            except Exception as e:
                duration = time.time() - start_time
                ACCESS_LOGGER.error({
                    "request_id": request_id,
                    "conversation_id": conv_id,
                    "endpoint": request.path,
                    "type": endpoint_type,
                    "status": "failed",
                    "error": str(e),
                    "duration": round(duration, 3)
                })
                SYSTEM_LOGGER.error(f"[{request_id}] ERROR {request.path} | {str(e)}")
                raise e
        return wrapper
    return decorator


def timing_decorator(label=None):
    """Decorator to measure execution time using perf_counter for internal functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            log_label = label if label else func.__name__
            SYSTEM_LOGGER.info(f"{log_label} completed in {elapsed_time:.6f} seconds")
            return result
        return wrapper
    return decorator
