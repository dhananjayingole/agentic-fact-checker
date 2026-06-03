"""
run.py - Application entry point.

Usage:
    python run.py

Or with custom settings:
    APP_PORT=9000 APP_DEBUG=true python run.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
# Ensure project root is on sys.path so `src` is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()  # Load .env file if present

import uvicorn
from loguru import logger

# Configure Loguru
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    level=LOG_LEVEL,
    colorize=True,
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    backtrace=True,
)

if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    debug = os.getenv("APP_DEBUG", "false").lower() == "true"

    logger.info(f"Starting server on http://{host}:{port}")
    logger.info(f"Swagger docs: http://localhost:{port}/docs")
    logger.info(f"Debug mode: {debug}")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level=LOG_LEVEL.lower(),
    )