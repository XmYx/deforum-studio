# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/deforum/docutils'))
project = 'deforum'
copyright = '2024, deforum studio'
author = 'deforum studio'
release = '0.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Handles Python-specific documentation features like automodule
    # 'sphinx_deforumdoc',  # Handles Python-specific documentation features like automodule
    'sphinx.ext.viewcode',  # Optionally, adds links to source code
    'sphinx_rtd_theme',
    # Add other necessary extensions here
]
templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'canonical_url': '',  # Set this if you have a preferred URL
    'analytics_id': '',   # Google Analytics ID if you have one
    'logo_only': False,   # True if you want to display only the logo in the sidebar
    'display_version': True,  # Whether to show the documentation version
    'prev_next_buttons_location': 'bottom',  # Location of the Next and Previous buttons
    'style_external_links': True,  # Style external links differently
    'vcs_pageview_mode': 'blob',  # How to display links to the repo ('blob', 'edit', or 'raw')
    'style_nav_header_background': 'white',  # Customize the navigation header background color
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 10,  # Adjust this as needed
    'includehidden': True,
    'titles_only': False
    # Add any other theme options you need
}


html_static_path = ['_static']
