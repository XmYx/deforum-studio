from sphinx.application import Sphinx

def setup(app: Sphinx):
    app.connect('autodoc-skip-member', skip_undecorated_functions)

def skip_undecorated_functions(app, what, name, obj, skip, options):
    """Skip documenting functions that do not have the @deforumdocs decorator."""
    return skip or not getattr(obj, '_include_in_deforumdocs', False)