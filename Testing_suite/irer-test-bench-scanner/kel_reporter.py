"""
kel_reporter.py
Aggregates scan results and generates executive summaries for the IRER Test Bench KEL system.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, List

class KelReporter:
    def __init__(self, results_path: str, summary_path: str):
        self.results_path = results_path
        self.summary_path = summary_path

    def load_results(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        with open(self.results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Adapt to new contract: dict with scan_meta, file_tree, axiom_states, metrics
        if isinstance(data, dict) and all(k in data for k in ("scan_meta", "file_tree", "axiom_states", "metrics")):
            # Flatten axiom_states for summary
            return data["axiom_states"]
        if isinstance(data, dict):
            data = [data]
        return data

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_violations": 0,
            "categories": {},
            "top_remedies": [],
            "notable_files": [],
        }
        remedy_counter = {}
        file_counter = {}
        for entry in results:
            category = entry.get("category", "Uncategorized")
            summary["categories"].setdefault(category, 0)
            summary["categories"][category] += 1
            summary["total_violations"] += 1
            remedies = entry.get("remedies", [])
            for remedy in remedies:
                remedy_counter[remedy] = remedy_counter.get(remedy, 0) + 1
            file = entry.get("file")
            if file:
                file_counter[file] = file_counter.get(file, 0) + 1
        summary["top_remedies"] = sorted(remedy_counter.items(), key=lambda x: -x[1])[:5]
        summary["notable_files"] = sorted(file_counter.items(), key=lambda x: -x[1])[:5]
        return summary

    def write_summary(self, summary: Dict[str, Any]):
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    def generate_markdown(self, summary: Dict[str, Any]) -> str:
        md = [f"# IRER Test Bench KEL Executive Summary\n", f"**Generated:** {summary['timestamp']}\n"]
        md.append(f"\n## Total Violations: {summary['total_violations']}\n")
        md.append("\n## Violations by Category:\n")
        for cat, count in summary["categories"].items():
            md.append(f"- **{cat}**: {count}")
        md.append("\n## Top Remedies:\n")
        for remedy, count in summary["top_remedies"]:
            md.append(f"- {remedy}: {count} occurrences")
        md.append("\n## Notable Files:\n")
        for file, count in summary["notable_files"]:
            md.append(f"- {file}: {count} issues")
        return "\n".join(md)

    def write_markdown(self, md: str):
        md_path = os.path.splitext(self.summary_path)[0] + ".md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md)

    def run(self):
        results = self.load_results()
        summary = self.summarize(results)
        self.write_summary(summary)
        md = self.generate_markdown(summary)
        self.write_markdown(md)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate KEL executive summary from scan results.")
    parser.add_argument('--results', type=str, default="reports/scan_results.json", help="Path to scan results JSON.")
    parser.add_argument('--summary', type=str, default="reports/scan_summary.json", help="Path to output summary JSON.")
    args = parser.parse_args()
    reporter = KelReporter(args.results, args.summary)
    reporter.run()
    print(f"Summary written to {args.summary} and markdown report generated.")
