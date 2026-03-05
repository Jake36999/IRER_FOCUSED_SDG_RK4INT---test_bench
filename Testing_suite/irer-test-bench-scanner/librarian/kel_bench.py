import ast
import sys
import os
import argparse
from typing import List, Dict, Set
from termcolor import colored  # pip install termcolor

# Import the Interface we built in Phase 2
try:
    from kel_interface import KelInterface
except ImportError:
    sys.exit("❌ Critical Error: Could not import 'kel_interface'. Make sure you are running this from the 'librarian' directory.")

class StaticAnalyzer(ast.NodeVisitor):
    """
    Parses Python code into an AST to find 'Risk Zones' without executing the code.
    """
    def __init__(self):
        super().__init__()
        self.risk_zones = []
        self.symbol_table = {}
        self.call_graph = {}
        self.imports = set()
        self.current_function = None
        self.axiom_violations = []  # Track structured axiom failures

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = None

    def visit_Call(self, node):
        """
        Detects calls to known JAX primitives.
        """
        if isinstance(node.func, ast.Attribute):
            # checking for things like jax.lax.scan
            full_name = self._get_full_attr_name(node.func)
            
            if "scan" in full_name or "while_loop" in full_name or "cond" in full_name:
                self.risk_zones.append({
                    "type": "JAX_PRIMITIVE",
                    "name": full_name,
                    "line": node.lineno,
                    "context": self.current_function or "global_scope",
                    "args": [a.arg for a in node.keywords] if hasattr(node, "keywords") else []
                })
        self.generic_visit(node)

    def _get_full_attr_name(self, node):
        """Helper to reconstruct 'jax.lax.scan' from AST nodes."""
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._get_full_attr_name(node.value)}.{node.attr}"
        except:
            return "unknown_call"
        return "unknown_call"

class KelBenchScanner:
    def __init__(self):
        self.interface = KelInterface()
        print(colored("\n🔭 KEL BENCH SCANNER | Sovereign Auditor Active", "cyan", attrs=["bold"]))
        print(colored(f"   Connected to Knowledge Base: {self.interface.stats()['vector_entries']} vectors loaded.\n", "cyan"))

    def scan_target(self, file_path: str):
        """
        Main execution flow.
        """
        if not os.path.exists(file_path):
            print(colored(f"❌ Error: File not found: {file_path}", "red"))
            return

        print(f"📄 Scanning Target: {colored(os.path.basename(file_path), 'yellow')}")
        
        # 1. Static Analysis
        with open(file_path, "r", encoding="utf-8") as source:
            tree = ast.parse(source.read())
        
        analyzer = StaticAnalyzer()
        analyzer.visit(tree)

        # Structured axiom reporting for each risk zone
        for zone in analyzer.risk_zones:
            self._apply_hard_rules(zone, analyzer)

        # 2. Semantic Querying
        if not analyzer.risk_zones:
            print(colored("   ✅ Clean Scan. No obvious JAX risk zones detected.", "green"))
            return

        print(colored(f"   ⚠️  Detected {len(analyzer.risk_zones)} Risk Zones. Consulting Librarian...", "yellow"))
        
        for zone in analyzer.risk_zones:
            self._evaluate_risk(zone, file_path)


    def _apply_hard_rules(self, zone: dict, analyzer):
        if zone['name'] == "jax.lax.scan":
            if "xs" in zone['args'] and "init" in zone['args']:
                pass # Valid
            else:
                # Emitting structured data instead of just printing
                analyzer.axiom_violations.append({
                    "axiom_id": 2,
                    "status": "FAIL",
                    "rule": "Sanitization & Schematic Rules",
                    "message": "'jax.lax.scan' missing explicit 'init' state.",
                    "line": zone.get('line', 0)
                })

    def _evaluate_risk(self, zone: Dict, file_path: str):
        """
        Queries the KEL database for historical context on this specific code pattern.
        """
        # Apply hard rules before querying the interface (no print)
        # No analyzer available in this context; skip axiom violation appending
        pass

        # Construct a query based on the Code Context
        query_context = f"Using {zone['name']} in function {zone['context']}."
        query_trace = f"Potential instability in {zone['name']} configuration."

        # Query the Interface (Limit to top 1 most relevant result)
        remedies = self.interface.query_remedies(
            error_trace=query_trace,
            context=query_context,
            limit=1
        )
        # (Optional: print or log for CLI, but not for contract-driven output)

def main():
    parser = argparse.ArgumentParser(description="KEL-Bench: Agnostic Physics Scanner")
    parser.add_argument("target", help="Path to the python file to scan")
    args = parser.parse_args()

    scanner = KelBenchScanner()
    scanner.scan_target(args.target)

if __name__ == "__main__":
    main()