def deforumdoc(func):
    """Decorator to mark functions to include in Sphinx documentation."""
    func._include_in_docs = True
    return func