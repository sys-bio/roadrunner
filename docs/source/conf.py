# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

# General information about the project.
project = u'libRoadRunner'
copyright = u'2013-2018, Andy Somogyi, J Kyle Medley, Kiri Choi, Herbert Suaro'
author = "Andy Somogyi, J Kyle Medley, Kiri Choi, Herbert Suaro"

# The full version, including alpha/beta/rc tags
release = '1.1.16'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    "sphinx.ext.autosectionlabel",
    "breathe",
    "sphinxcontrib.bibtex",
    "sphinx.ext.doctest",
    "sphinx_tabs.tabs",
    "sphinx_rtd_theme",
    # "PyExec"
    # "sphinx_execute_code" #didnt work
]

# make todo items show in docs
todo_include_todos = True


# Breathe Configuration
breathe_default_project = "roadrunner"
import os
DOCS_SOURCE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.dirname(DOCS_SOURCE_DIR)
DOXYGEN_OUTPUT_DIR = os.path.join(DOCS_DIR, "doxygen-output")
DOXYGEN_XML_OUTPUT_DIR = os.path.join(DOXYGEN_OUTPUT_DIR, 'xml')
breathe_projects = {
    "roadrunner": DOXYGEN_XML_OUTPUT_DIR
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
