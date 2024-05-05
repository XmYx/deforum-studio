from sphinx.application import Sphinx

def setup(app: Sphinx):
    app.connect('autodoc-skip-member', skip_member)

def skip_member(app, what, name, obj, skip, options):
    # Skip if not decorated with `@deforumdoc`
    return skip or not getattr(obj, '_include_in_docs', False)