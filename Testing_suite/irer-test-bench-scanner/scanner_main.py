
import argparse
import os
from core.file_walker import FileWalker
from core.ast_parser import ASTParser
from core.ai_orchestrator import AIOrchestrator
from librarian.kel_bench import StaticAnalyzer

def main():
    parser = argparse.ArgumentParser(description="IRER Test Bench Scanner")
    parser.add_argument('--path', type=str, required=True, help='Path to the directory to scan')
    parser.add_argument('--mode', type=str, choices=['full', 'quick'], default='full', help='Scan mode')
    parser.add_argument('--lmstudio-url', type=str, required=True, help='URL for the LLM Studio integration')
    parser.add_argument('--ai-persona', type=str, required=True, help='AI persona for analysis')
    parser.add_argument('--rules', type=str, required=True, help='Path to governance rules YAML file')
    parser.add_argument('--force-fresh-scan', action='store_true', help='Force a fresh scan regardless of previous results')
    parser.add_argument('--output-path', type=str, default=None, help='Explicit output path for scan_results.json')

    args = parser.parse_args()


    file_walker = FileWalker()
    ast_parser = ASTParser()
    ai_orchestrator = AIOrchestrator()

    if args.force_fresh_scan:
        print("Forcing a fresh scan...")

    print(f"Scanning directory: {args.path}")
    files_to_scan = file_walker.scan_directory(args.path)

    import json
    from datetime import datetime

    file_tree = files_to_scan
    all_axiom_violations = []
    total_files = len(files_to_scan)
    red_tier_count = 0
    ast_complexities = []
    for file in files_to_scan:
        print(f"Parsing file: {file}")
        # Use StaticAnalyzer for structured axiom reporting
        import ast
        with open(file, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=file)
        analyzer = StaticAnalyzer()
        analyzer.visit(tree)
        # Collect axiom violations for God View
        for v in analyzer.axiom_violations:
            all_axiom_violations.append(v)
        # Optionally, collect complexity metrics (stubbed as 1 per file)
        ast_complexities.append(1)
        # Optionally, run AI orchestrator (can be extended for insights)
        # insights = ai_orchestrator.run_analysis({'ast': tree, 'violations': analyzer.axiom_violations})
        # if insights:
        #     print(f"  AI Insights: {insights}")

    avg_complexity = sum(ast_complexities) / total_files if total_files else 0
    # Use the last analyzer's axiom_violations if available, else []
    final_report = {
        "scan_meta": {
            "timestamp": datetime.now().isoformat(),
            "target_path": args.path,
        },
        "file_tree": file_tree,
        "axiom_states": analyzer.axiom_violations if 'analyzer' in locals() else [],
        "metrics": {
            "total_files_scanned": total_files,
            "red_tier_count": red_tier_count,
            "ast_complexity_average": avg_complexity
        }
    }
    output_path = args.output_path or os.path.join("reports", "scan_results.json")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nScan results written to {output_path}")

if __name__ == "__main__":
    main()