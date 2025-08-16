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
    description="AI-powered document processing and summarization with Streamlit + FastAPI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ElevateAI Team",
    author_email="team@elevateai.com",
    url="https://github.com/TwinkleDreamm/ElevateAI_SummarizeApplication",
    packages=find_packages(where=".", include=["src*", "config*"]),
    package_dir={"": "."},
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
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
        ],
        "full": [
            "moviepy>=1.0.3",  # Video processing
            "torch>=2.0.0",    # Advanced ML models
        ],
    },
    entry_points={
        "console_scripts": [
            "elevateai=main:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/TwinkleDreamm/ElevateAI_SummarizeApplication/issues",
        "Source": "https://github.com/TwinkleDreamm/ElevateAI_SummarizeApplication",
        "Documentation": "https://github.com/TwinkleDreamm/ElevateAI_SummarizeApplication/wiki",
    },
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json", "*.toml"],
        "config": ["*.py"],
        "data": [".gitkeep"],
    },
    zip_safe=False,
)
