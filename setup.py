"""
Setup configuration for Document Classification API
Author: Bachir Fahmi
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="document-classification-api",
    version="1.0.0",
    author="Bachir Fahmi",
    author_email="bachir.fahmi@example.com",
    description="Production-ready document classification system using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bachirfahmi/document-classification-api",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "doc-classifier=app:main",
        ],
    },
    keywords="document classification, machine learning, OCR, AI, FastAPI",
    project_urls={
        "Bug Reports": "https://github.com/bachirfahmi/document-classification-api/issues",
        "Source": "https://github.com/bachirfahmi/document-classification-api",
        "Documentation": "https://github.com/bachirfahmi/document-classification-api#readme",
    },
)