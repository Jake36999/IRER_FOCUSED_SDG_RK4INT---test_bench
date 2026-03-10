import os
import ast
from collections import defaultdict

def find_py_files(root):
    py_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.py'):
                py_files.append(os.path.join(dirpath, f))
    return py_files

def extract_defs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)
    defs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = max(getattr(node, 'end_lineno', start), start)
            code = '\n'.join(source.splitlines()[start-1:end])
            defs.append((node.name, start, end, code))
    return defs

def detect_duplicates(root):
    files = find_py_files(root)
    code_map = defaultdict(list)
    for file in files:
        for name, start, end, code in extract_defs(file):
            code_key = hash(code)
            code_map[code_key].append((file, name, start, end))
    # Report duplicates
    for code_key, locations in code_map.items():
        if len(locations) > 1:
            print("Duplicate code found in:")
            for file, name, start, end in locations:
                print(f"  {file}: {name} (lines {start}-{end})")
            print()

if __name__ == "__main__":
    detect_duplicates(".")
