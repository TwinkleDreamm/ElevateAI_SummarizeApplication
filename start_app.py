#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple startup script for ElevateAI that ensures conda Project_1 environment.
"""

import sys
import os
import subprocess
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    import codecs
    # Set environment variables for UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

    # Try to reconfigure stdout/stderr for UTF-8
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        # Fallback for older Python versions
        try:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        except Exception:
            pass


def check_dependencies():
    """Check if key dependencies are available."""
    try:
        import streamlit
        import openai
        import langchain
        print("✅ Key dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("\nPlease install dependencies:")
        print("   conda activate Project_1")
        print("   pip install -r requirements.txt")
        return False

def start_streamlit_app():
    """Start the Streamlit application."""
    try:
        # Use the main app entry point
        app_path = Path(__file__).parent / "src" / "interface" / "app.py"
        
        print("🚀 Starting ElevateAI Streamlit application...")
        print("📱 Open your browser to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 50)
        
        # Start Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)

def main():
    """Main startup function."""
    print("🚀 Starting ElevateAI Application")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        print("⚠️ Some dependencies missing, but continuing...")

    print("\n" + "=" * 40)

    # Start the application
    start_streamlit_app()

if __name__ == "__main__":
    main()
