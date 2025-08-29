# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
# Add the project root and the src directory so Python can import the package
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

project = 'RAG, search and math agent'
copyright = '2025, Alessio Buda'
author = 'Alessio Buda'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # per Google/NumPy style
    "sphinx.ext.autodoc.typehints",
]
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Avoid hard import failures during autodoc by mocking heavy/optional deps
autodoc_mock_imports = [
    "crewai",
    "crewai_tools",
    "langchain",
    "langchain_community",
    "langchain_openai",
    "faiss",
    "duckduckgo_search",
    "bs4",
    "dotenv",
]