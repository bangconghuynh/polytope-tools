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
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'polytope-tools'
copyright = '2019, Bang C. Huynh'
author = 'Bang C. Huynh'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
import sphinx_rtd_theme
extensions = [\
        'sphinx.ext.autodoc',\
        'sphinx.ext.napoleon',\
        'sphinx.ext.intersphinx',\
        'sphinx_autodoc_typehints',\
        'sphinx.ext.coverage',\
        'sphinx.ext.autosummary',\
        'sphinxcontrib.bibtex',\
        'sphinx.ext.inheritance_diagram',\
        'sphinx_rtd_theme'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
highlight_language = 'python3'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# -- Options for autodoc -----------------------------------------------------

autodoc_member_order = 'bysource'
autodoc_default_options = {'members': True}
autoclass_content = 'both'

# -- Options for autosummary -------------------------------------------------

autosummary_generate = True
autosummary_imported_members = True

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {'python': ('http://docs.python.org/3/', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy/', None)}
