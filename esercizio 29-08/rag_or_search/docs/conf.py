# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
sys.path.insert(0, os.path.abspath(".."))

project = 'RAG, search and solve math'
copyright = '2025, Maci, Giacobbe, Buda, Meziane'
author = 'Maci, Giacobbe, Buda, Meziane'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

extensions = ["sphinx.ext.autodoc","sphinx.ext.napoleon",  # per Google/NumPy style
"sphinx.ext.autodoc.typehints",]

napoleon_google_docstring = True
napoleon_numpy_docstring = False