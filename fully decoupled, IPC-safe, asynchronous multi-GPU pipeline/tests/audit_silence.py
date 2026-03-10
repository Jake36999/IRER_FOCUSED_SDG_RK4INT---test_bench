"""
tools/audit_silence.py
MANDATE: Scan codebase for 'Black Holes' (swallowed errors) and 'Silent Ejects'.
"""
import ast
import os
import sys

class SilenceHunter(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.errors = []
        self.has_jax_import = False
        self.jax_debug_enabled = False

    def visit_Import(self, node):
        for alias in node.names:
            if 'jax' in alias.name:
                self.has_jax_import = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and 'jax' in node.module:
            self.has_jax_import = True
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        # 1. The Black Hole: Empty except blocks
        if not node.body:
            self.errors.append(f"Line {node.lineno}: [CRITICAL] Empty 'except' block.")
            return

        # 2. The Mute Pass: 'except: pass'
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.errors.append(f"Line {node.lineno}: [CRITICAL] 'except: pass' detected. Error swallowed.")
            return

        # 3. The Fake Log: Catching error without raising or exiting
        # We look for 'raise', 'return', 'sys.exit', or assignment to a status variable
        has_raise = any(isinstance(n, ast.Raise) for n in node.body)
        has_return = any(isinstance(n, ast.Return) for n in node.body)
        has_exit = False
        
        for n in node.body:
            if isinstance(n, ast.Call):
                # Check for sys.exit() or explicit status updates
                if isinstance(n.func, ast.Attribute) and n.func.attr == 'exit':
                    has_exit = True
                # Check for print/logging (Weak mitigation, but better than nothing)
        
        if not (has_raise or has_return or has_exit):
            self.errors.append(f"Line {node.lineno}: [RISK] Error caught without Raise/Return/Exit. Verify logic.")

    def visit_Call(self, node):
        # 4. JAX Debugging Check
        # Looks for jax.config.update("jax_debug_nans", True)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'update':
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "jax_debug_nans":
                self.jax_debug_enabled = True

    def finalize(self):
        if self.has_jax_import and not self.jax_debug_enabled:
             self.errors.append("Line 1: [CONFIG] JAX imported but 'jax_debug_nans' not enabled.")

def scan_directory(path):
    print(f"--- 🛡️  Starting Silence Audit on {path} ---")
    risk_found = False
    
    for root, dirs, files in os.walk(path):
        if "venv" in root or "__pycache__" in root:
            continue
            
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                        hunter = SilenceHunter(full_path)
                        hunter.visit(tree)
                        hunter.finalize()
                        
                        if hunter.errors:
                            risk_found = True
                            print(f"\n📄 {file}:")
                            for err in hunter.errors:
                                print(f"  ❌ {err}")
                    except SyntaxError as e:
                        print(f"\n⚠️  Syntax Error in {file}: {e}")

    if not risk_found:
        print("\n✅ Clean Audit. No black holes detected.")
    else:
        print("\n🚩 Risks detected. Review above.")

if __name__ == "__main__":
    scan_directory(".")