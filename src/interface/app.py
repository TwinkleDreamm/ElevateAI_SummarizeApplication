#!/usr/bin/env python3
"""
Main entry point for ElevateAI Streamlit multi-page application.
"""

import streamlit as st

def main():
    st.set_page_config(
        page_title="ElevateAI Notebooks",
        page_icon="📓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📓 ElevateAI Notebooks")
    st.markdown("Welcome to your knowledge management system!")
    
    st.info("""
    This is the main entry point for the ElevateAI application.
    
    Please use the sidebar navigation to access different pages:
    - **Home**: View and manage your notebooks
    - **Create Notebook**: Create a new notebook
    - **Notebook**: View and chat with a specific notebook
    """)

if __name__ == "__main__":
    main()
