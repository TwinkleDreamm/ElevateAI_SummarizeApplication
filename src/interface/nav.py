"""
Navigation helper for Streamlit pages.
"""

import streamlit as st
import subprocess
import sys
from pathlib import Path


def switch_to_page(page_name: str):
    """Switch to a specific page by running it directly."""
    try:
        # Get the current script path
        current_script = Path(__file__).parent / "pages" / f"{page_name}.py"
        
        if current_script.exists():
            # Run the page script
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(current_script)])
        else:
            st.error(f"Page {page_name} not found")
    except Exception as e:
        st.error(f"Failed to switch to page {page_name}: {e}")


def go_to_notebook(notebook_id: str):
    """Go to notebook page with specific notebook ID."""
    st.session_state["current_notebook_id"] = notebook_id
    switch_to_page("03_Notebook")


def go_to_create_notebook():
    """Go to create notebook page."""
    switch_to_page("02_CreateNotebook")


def go_to_home():
    """Go to home page."""
    switch_to_page("01_Home")


