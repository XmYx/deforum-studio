# Generate .rst files from your Python source code

pip install sphinx sphinx-rtd-theme

#sphinx-apidoc -f -o docs/source/modules src/deforum

# Build the HTML documentation
sphinx-build -b html docs/source docs/build

# Deactivate virtual environment if previously activated
# deactivate

echo "Documentation generated at docs/build"