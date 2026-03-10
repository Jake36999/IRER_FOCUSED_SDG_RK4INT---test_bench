import ast
import os
import sys

class AxiomaticAlignmentVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_Attribute(self, node):
        # 1. Detect Legacy V10 Errors: np.gradient
        if isinstance(node.value, ast.Name):
            if node.value.id in ['np', 'numpy'] and node.attr == 'gradient':
                self.violations.append(f"Line {node.lineno}: BANNED FUNCTION - 'np.gradient' used. Must use Spectral Laplacians.")
        self.generic_visit(node)

    def visit_For(self, node):
        # 2. Check for JAX purity ONLY inside JIT-compiled functions
        parent = getattr(node, 'parent_function', None)
        if parent:
            for decorator in parent.decorator_list:
                # Check for @jax.jit or @jit
                if (isinstance(decorator, ast.Attribute) and decorator.attr == 'jit') or \
                   (isinstance(decorator, ast.Name) and decorator.id == 'jit'):
                    self.violations.append(f"Line {node.lineno}: WARNING - Python 'for' loop inside @jax.jit detected. Must use jax.lax.scan for JIT purity.")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Tag all children with their parent function so we can check decorators
        for child in ast.walk(node):
            child.parent_function = node
        self.generic_visit(node)

def audit_codebase(directory: str = "."):
    print("[Axiomatic Lens] Scanning codebase for compliance...")
    target_files = [f for f in os.listdir(directory) if f.endswith('.py')]
    
    total_violations = 0
    for filename in target_files:
        with open(filename, "r", encoding="utf-8") as source:
            try:
                tree = ast.parse(source.read())
            except Exception as e:
                print(f"❌ {filename} failed to parse: {e}")
                continue
            
        visitor = AxiomaticAlignmentVisitor()
        visitor.visit(tree)
        
        if visitor.violations:
            print(f"\n❌ {filename} failed Axiomatic Audit:")
            for violation in visitor.violations:
                if "BANNED" in violation:
                    print(f"   [FATAL] {violation}")
                    total_violations += 1
                else:
                    print(f"   [WARN]  {violation}")
    
    if total_violations > 0:
        print("\n[Axiomatic Lens] PRE-FLIGHT HALTED. Code violates IRER constitutional mandates.")
        sys.exit(1)
    else:
        print("\n✅ [Axiomatic Lens] Codebase compliant. Cleared for execution.")

if __name__ == "__main__":
    audit_codebase()