"""
silence_hunter.py
Purpose: Static Analysis to find 'Silent Failures' (swallowed exceptions) in the backend.
Adapted from Directory_bundler_v4.5 logic and workspace_packager_v2.3 filtering.
"""
import ast
import os
import sys

# --- Configuration (Extracted from workspace_packager_v2.3.py) ---
IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", 
    ".idea", ".vscode", "dist", "build", "coverage"
}

class TerminalUI:
    FAIL = '\033[91m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

class SilenceDetector(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.issues = []
        self.has_jax = False
        self.jax_debug_enabled = False

    def visit_Import(self, node):
        for name in node.names:
            if 'jax' in name.name:
                self.has_jax = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and 'jax' in node.module:
            self.has_jax = True
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """
        Visits every 'except:' block in the code.
        """
        # 1. Check for 'pass' only (The Black Hole)
        # Matches: except: pass
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.issues.append({
                "line": node.lineno,
                "type": "CRITICAL",
                "msg": "Empty 'except' block with 'pass'. Errors are being swallowed."
            })
            return

        # 2. Check for missing 'raise', 'return', or 'exit' (The Zombie Process)
        # Matches: except: print("Error") -> Continues execution in broken state
        has_raise = any(isinstance(n, ast.Raise) for n in node.body)
        has_return = any(isinstance(n, ast.Return) for n in node.body)
        has_exit = False
        
        # Scan for sys.exit(), exit(), or specific status assignments
        for child in node.body:
            if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                func_name = ""
                # Handle module.func() like sys.exit()
                if isinstance(child.value.func, ast.Attribute):
                    func_name = child.value.func.attr
                # Handle direct func() like exit()
                elif isinstance(child.value.func, ast.Name):
                    func_name = child.value.func.id
                
                if "exit" in func_name:
                    has_exit = True

        # If it doesn't stop the flow (Raise/Return/Exit), it's a risk.
        if not (has_raise or has_return or has_exit):
            self.issues.append({
                "line": node.lineno,
                "type": "WARNING",
                "msg": "Exception caught but execution continues (No raise/return/exit)."
            })

    def visit_Call(self, node):
        """
        Detects if JAX debugging is explicitly enabled.
        """
        # Look for jax.config.update("jax_debug_nans", True)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'update':
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "jax_debug_nans":
                self.jax_debug_enabled = True
        self.generic_visit(node)

def scan_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        detector = SilenceDetector(filepath)
        detector.visit(tree)
        
        # Post-Scan Checks
        if detector.has_jax and not detector.jax_debug_enabled:
            detector.issues.append({
                "line": 1,
                "type": "CONFIG",
                "msg": "JAX imported but 'jax_debug_nans' not enabled. Recommended for physics debugging."
            })
        
        if detector.issues:
            print(f"\n{TerminalUI.BOLD}File: {filepath}{TerminalUI.ENDC}")
            for issue in detector.issues:
                color = TerminalUI.FAIL if issue['type'] == "CRITICAL" else TerminalUI.WARNING
                if issue['type'] == "CONFIG": color = TerminalUI.GREEN
                print(f"  {color}[{issue['type']}] Line {issue['line']}: {issue['msg']}{TerminalUI.ENDC}")
            return True
                
    except Exception as e:
        print(f"{TerminalUI.WARNING}Skipping {filepath}: {e}{TerminalUI.ENDC}")
    return False

def main():
    target_dir = "."
    print(f"--- 🛡️  Silence Hunter Active in: {os.path.abspath(target_dir)} ---")
    issues_found = 0
    
    for root, dirs, files in os.walk(target_dir):
        # Apply ignore logic from workspace_packager_v2.3.py
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file.endswith(".py"):
                if scan_file(os.path.join(root, file)):
                    issues_found += 1

    if issues_found == 0:
        print(f"\n{TerminalUI.GREEN}✅ Codebase is clean. No silent failures detected.{TerminalUI.ENDC}")
    else:
        print(f"\n{TerminalUI.FAIL}🚩 Scan Complete. Potential risks found in {issues_found} files.{TerminalUI.ENDC}")

if __name__ == "__main__":
    main()