import json
import logging
import logging.config
import os
import sys
from datetime import datetime
from typing import Any

from app.shared.config import settings
from app.shared.context import get_request_id


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for log records.
    Inspired by Monolog's JsonFormatter.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": get_request_id(),
        }
        
        # Add exception info if it exists
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields (from record.__dict__)
        # We skip standard LogRecord attributes
        standard_attrs = {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "message", "module",
            "msecs", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName", "request_id"
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_record[key] = value
            
        return json.dumps(log_record)


class RequestIDFilter(logging.Filter):
    """
    Filter that adds request_id to log records.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def setup_logging() -> None:
    """
    Set up advanced logging configuration.
    Inspired by Monolog in Symfony.
    """
    log_level = settings.LOG_LEVEL.upper()
    log_format = settings.LOG_FORMAT.upper()
    
    # Define the logging configuration
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "request_id": {
                "()": "app.shared.logging.RequestIDFilter",
            },
        },
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": "app.shared.logging.JSONFormatter",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json" if log_format == "JSON" else "standard",
                "stream": sys.stdout,
                "filters": ["request_id"],
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console"],
                "level": log_level,
                "propagate": True,
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "app": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
        },
    }
    
    # Add file handler if LOG_FILE is configured
    if settings.LOG_FILE:
        log_file_path = settings.LOG_FILE
        # Ensure directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json" if log_format == "JSON" else "standard",
            "filename": log_file_path,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
            "filters": ["request_id"],
        }
        # Add file handler to all loggers
        config["loggers"][""]["handlers"].append("file")
        config["loggers"]["uvicorn"]["handlers"].append("file")
        config["loggers"]["app"]["handlers"].append("file")
        
    logging.config.dictConfig(config)
