"""
Setup script for ElevateAI application.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="elevateai",
    version="1.0.0",
    description="AI-powered content analysis and summarization application with memory system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ElevateAI Team",
    author_email="team@elevateai.com",
    url="https://github.com/elevateai/elevateai",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Multimedia :: Video",
        "Topic :: Office/Business :: Office Suites",
    ],
    keywords="ai, summarization, nlp, text analysis, memory system, streamlit, langchain",
    entry_points={
        "console_scripts": [
            "elevateai=main:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/elevateai/elevateai/issues",
        "Source": "https://github.com/elevateai/elevateai",
        "Documentation": "https://github.com/elevateai/elevateai/wiki",
    },
)
