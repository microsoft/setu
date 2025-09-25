# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add the project root to the Python path for autodoc
sys.path.insert(0, os.path.abspath(".."))

project = "Setu"
copyright = "(c) 2025 Vajra Team; Georgia Institute of Technology; Microsoft Corporation."
author = "2025 Vajra Team; Georgia Institute of Technology; Microsoft Corporation"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "breathe",
    "myst_parser",
]

# -- Autodoc configuration --------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Generate autosummary even if no references
autosummary_generate = True
autosummary_imported_members = True

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Breathe configuration --------------------------------------------------
breathe_projects = {"setu": "doxygen_output/xml"}
breathe_default_project = "setu"

# -- MyST configuration -----------------------------------------------------
myst_enable_extensions = [
    "deflist",
    "tasklist",
    "dollarmath",
    "amsmath",
    "colon_fence",
    "attrs_inline",
    "attrs_block",
]

# Enable MyST for all files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Allow Sphinx directives in Markdown
myst_enable_extensions.append("substitution")
myst_heading_anchors = 3

# Configure MyST to parse directives in markdown files
myst_update_mathjax = False
myst_dmath_double_inline = True
myst_dmath_allow_labels = True
myst_dmath_allow_space = True
myst_dmath_allow_digits = True
myst_amsmath_enable = True
myst_enable_extensions.append("dollarmath")
myst_enable_extensions.append("amsmath")

# Enable direct use of Sphinx directives in Markdown
myst_enable_extensions.append("colon_fence")
myst_enable_extensions.append("linkify")
myst_enable_extensions.append("smartquotes")
myst_enable_extensions.append("replacements")
myst_enable_extensions.append("html_admonition")
myst_enable_extensions.append("html_image")
myst_enable_extensions.append("fieldlist")

# -- Intersphinx configuration ----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "site", "doxygen_output"]

# Use index.md as the root document
root_doc = "index"
master_doc = "index"

# -- Options for HTML output ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"{project} Documentation"

html_theme_options = {
    "sidebar_hide_name": False,
    "source_repository": "https://github.com/project-vajra/setu/",
    "source_branch": "main",
    "source_directory": "docs/",
    "announcement": "🚀 Setu is in active development - API may change between versions",
    "light_logo": "images/setu-logo-light.png",
    "dark_logo": "images/setu-logo-dark.png",
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/project-vajra/setu",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
        {
            "name": "Discord",
            "url": "https://discord.gg/wjaSvGgsNN",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path d="M13.545 2.907a13.227 13.227 0 0 0-3.257-1.011.05.05 0 0 0-.052.025c-.141.25-.297.577-.406.833a12.19 12.19 0 0 0-3.658 0 8.258 8.258 0 0 0-.412-.833.051.051 0 0 0-.052-.025c-1.125.194-2.22.534-3.257 1.011a.041.041 0 0 0-.021.018C.356 6.024-.213 9.047.066 12.032c.001.014.01.028.021.037a13.276 13.276 0 0 0 3.995 2.02.05.05 0 0 0 .056-.019c.308-.42.582-.863.818-1.329a.05.05 0 0 0-.01-.059.051.051 0 0 0-.018-.011 8.875 8.875 0 0 1-1.248-.595.05.05 0 0 1-.02-.066.051.051 0 0 1 .015-.019c.084-.063.168-.129.248-.195a.05.05 0 0 1 .051-.007c2.619 1.196 5.454 1.196 8.041 0a.052.052 0 0 1 .053.007c.08.066.164.132.248.195a.051.051 0 0 1-.004.085 8.254 8.254 0 0 1-1.249.594.05.05 0 0 0-.03.03.052.052 0 0 0 .003.041c.24.465.515.909.817 1.329a.05.05 0 0 0 .056.019 13.235 13.235 0 0 0 4.001-2.02.049.049 0 0 0 .021-.037c.334-3.451-.559-6.449-2.366-9.106a.034.034 0 0 0-.02-.019Zm-8.198 7.307c-.789 0-1.438-.724-1.438-1.612 0-.889.637-1.613 1.438-1.613.807 0 1.45.73 1.438 1.613 0 .888-.637 1.612-1.438 1.612Zm5.316 0c-.788 0-1.438-.724-1.438-1.612 0-.889.637-1.613 1.438-1.613.807 0 1.451.73 1.438 1.613 0 .888-.631 1.612-1.438 1.612Z"/></svg>',
            "class": "",
        },
        {
            "name": "X",
            "url": "https://x.com/ProjectSetu",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path d="M12.6.75h2.454l-5.36 6.142L16 15.25h-4.937l-3.867-5.07-4.425 5.07H.316l5.733-6.57L0 .75h5.063l3.495 4.633L12.601.75Zm-.86 13.028h1.36L4.323 2.145H2.865l8.875 11.633Z"/></svg>',
            "class": "",
        },
    ],
}

html_static_path = ["_static"]
html_favicon = "_static/images/setu-favicon.png"

# Add custom CSS for logo sizing and other customizations
html_css_files = [
    "css/custom.css",
]

# -- Options for todo extension ---------------------------------------------
todo_include_todos = True

# -- Options for coverage extension -----------------------------------------
coverage_show_missing_items = True

# -- Custom configuration ---------------------------------------------------

# Mock imports for modules that aren't available during documentation build
autodoc_mock_imports = [
    "setu._native",
    "setu._native.core",
    "setu._native.core.datatypes",
    "setu._native.engine",
    "setu._native.llm",
    "setu._kernels",
    "torch",
    "transformers",
    "vllm",
    "vidur",
    "flashinfer",
    "cuda",
    "cupy",
    "triton",
    "zmq",
    "ddsketch",
    "tokenizers",
    "pyarrow",
    "tiktoken",
    "sentencepiece",
    "pytest",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
    "plotly",
    "altair",
    "bokeh",
    "holoviews",
    "hvplot",
    "datashader",
    "panel",
    "msgpack",
    "msgpack_numpy",
    "wandb",  # Mock wandb to avoid circular import issues
    "wandb.apis",
    "wandb.sdk",
    "wandb.data_types",
]


# Suppress specific warnings for known issues
import warnings
warnings.filterwarnings("ignore", message="duplicate object description")

# Function to run before each autodoc
def skip_if_missing(app, what, name, obj, would_skip, options):
    """Skip documentation for objects that can't be imported."""
    if would_skip:
        return True
    try:
        # Try to access the object to see if it exists
        _ = obj.__doc__
        return False
    except (AttributeError, ImportError):
        return True


def setup(app):
    app.connect("autodoc-skip-member", skip_if_missing)
