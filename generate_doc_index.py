import ast
import os

import fnmatch

def is_directory_included(directory, patterns):
    """Check if the directory matches any of the include patterns."""
    for pattern in patterns:
        if fnmatch.fnmatch(directory, pattern):
            return True
    return False
include_patterns = [
    'deforum*'  # Include all directories starting with 'deforum'
]
class DecoratorFinder(ast.NodeVisitor):
    def __init__(self):
        self.decorated_functions = []

    def visit_FunctionDef(self, node):
        """Visit each function and method to check for the decorator."""
        self.check_decorators(node)

    def visit_ClassDef(self, node):
        """Visit each class to find methods with decorators."""
        class_name = node.name
        for n in node.body:
            if isinstance(n, ast.FunctionDef):
                for decorator in n.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'deforumdoc':
                        self.decorated_functions.append(f"{class_name}.{n.name}")
                        break

    def check_decorators(self, node):
        """Check if a function/method node has the @deforumdoc decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'deforumdoc':
                self.decorated_functions.append(node.name)
                break




def process_file(file_path):
    """Return the names of functions in the file that have the @deforumdoc decorator."""
    with open(file_path, 'r') as file:
        source = file.read()
        tree = ast.parse(source, filename=file_path)
        finder = DecoratorFinder()
        finder.visit(tree)
        return finder.decorated_functions  # Return list of decorated function names



def generate_rst_files(source_directory, rst_directory):
    """Generate RST files for functions that are decorated with @deforumdoc."""
    os.makedirs(rst_directory, exist_ok=True)
    module_names = []
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)
                decorated_functions = process_file(file_path)
                if decorated_functions:  # Only process if there are decorated functions
                    module_name = os.path.splitext(file)[0]
                    module_path = os.path.relpath(root, source_directory).replace(os.sep, '.')
                    full_module_name = f"{module_path}.{module_name}" if module_path != '.' else module_name
                    module_names.append(full_module_name)  # Changed from replacing dots with underscores
                    rst_filename = f"{full_module_name.replace('.', '_')}.rst"
                    rst_filepath = os.path.join(rst_directory, rst_filename)

                    with open(rst_filepath, 'w') as rst_file:
                        rst_file.write(f"{full_module_name}\n")
                        rst_file.write("=" * len(full_module_name) + "\n\n")
                        rst_file.write(f".. automodule:: {full_module_name}\n")
                        rst_file.write("   :members: \n")# + ", ".join(decorated_functions) + "\n")  # Specify which members to include
                        rst_file.write("   :undoc-members:\n")
                        rst_file.write("   :show-inheritance:\n")
                    print(f"Generated RST for {full_module_name}")

    # Update index file generation to use dot notation
    with open(os.path.join(rst_directory, 'deforum_index.rst'), 'w') as index_file:
        index_file.write("DeForUM Modules\n")
        index_file.write("===============\n\n")
        index_file.write("This section contains documentation for all DeForUM modules.\n\n")
        index_file.write(".. toctree::\n")
        index_file.write("   :maxdepth: 2\n\n")
        for name in module_names:
            index_file.write(f"   {name.replace('.', '_')}\n")  # Changed from underscore to dot notation



# Usage example
generate_rst_files('src', 'docs/source/modules')
