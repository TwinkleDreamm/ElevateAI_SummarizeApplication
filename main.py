"""
Main entry point for ElevateAI application.
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
        # Ensure necessary directories exist
        settings.ensure_directories()
        
        # Log startup
        logger.info("Starting ElevateAI application")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Python path: {sys.path}")
        
        # Start Streamlit app
        streamlit_main()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
