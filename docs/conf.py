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
import glob
import json

sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../tfcomb'))


# -- Project information -----------------------------------------------------

project = 'TF-COMB'
copyright = '2021, Loosolab'
author = 'Loosolab'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              "nbsphinx",
              "nbsphinx_link",
              ]

napoleon_numpy_docstring = True
autodoc_member_order = 'bysource'

mathjax3_config = {'chtml': {'displayAlign': 'center'}}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = ['tfcomb.counting', 'numpy', 'matplotlib', 'qnorm', 'kneed', 'pandas', 'glob', 'scipy', 'sklearn',
                        'tobias', 'seaborn', 'networkx', 'community', 'goatools', 'statsmodels', 'mpl_toolkits', 'dill', 'pysam',
                        'pyBigWig', 'tqdm', 'IPython', 'graphviz', 'adjustText', 'psutil', 'uropa']

# -- Create nblink files  -------------------------------------------------

#Remove all previous .nblink files
links = glob.glob("examples/*.nblink")
for l in links:
    os.remove(l) 

#Create nblinks for current notebooks
notebooks = glob.glob("../examples/*.ipynb")
for f in notebooks:
    f_name = os.path.basename(f).replace(".ipynb", "")

    d = {"path": "../" + f} 
    with open("examples/" + f_name + ".nblink", 'w') as fp:
        json.dump(d, fp)

nbsphinx_execute = 'never'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_logo = "_figures/tfcomb_logo.png"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']