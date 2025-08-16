#!/usr/bin/env python3
"""
Start script for ElevateAI Streamlit application.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ.setdefault('PYTHONPATH', str(project_root))

try:
    from src.interface.streamlit_app import main as streamlit_main
    from src.utils.logger import logger
    from config.settings import settings
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)


def main():
    """Main application entry point."""
    try:
        logger.info("Starting ElevateAI Streamlit application")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Settings loaded from: {settings.project_root}")
        
        # Start Streamlit app
        streamlit_main()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
