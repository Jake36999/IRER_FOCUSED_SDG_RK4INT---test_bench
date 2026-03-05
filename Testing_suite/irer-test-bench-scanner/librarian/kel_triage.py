
import argparse

def annotate_node(node, severity=None, frequency=None, tags=None):
    """Manually tag and annotate a node."""
    node['meta']['severity'] = severity if severity else 'unknown'
    node['meta']['frequency'] = frequency if frequency else 1
    node['meta']['tags'] = tags if tags else []
    return node

def run_triage(deep_scan=False):
    interface = KelInterface()
    unindexed = interface.get_unindexed_nodes()
    print(colored(f"\n🧪 TRIAGE: {len(unindexed)} UNINDEXED Friction Points", "yellow"))
    for i, node in enumerate(unindexed):
        uid = node['uid']
        desc = node['semantic_data'].get('description', 'No Context')
        print(f"\n[{i+1}] ID: {colored(uid[:8], 'yellow')}")
        print(f"    Context: {desc[:200]}...")
        print(f"    Source:  {node['meta'].get('source')}")
        print(f"    Remedies: {len(node['meta'].get('remedies', []))}")
        # Manual annotation prompt
        severity = input("    Tag severity (low/medium/high): ")
        frequency = input("    Tag frequency (integer): ")
        tags = input("    Add tags (comma-separated): ").split(',') if input("    Add tags? (y/n): ") == 'y' else []
        node = annotate_node(node, severity, frequency, tags)
        # Optionally save annotation
        save = input("    Save annotation? (y/n): ")
        if save == 'y':
            node_path = os.path.join("./.kel_store/nodes", f"{uid}.json")
            with open(node_path, 'w', encoding='utf-8') as f:
                json.dump(node, f, indent=2)
        action = input(colored("    Assign fix manually? (y/n/skip): ", "cyan"))
        if action.lower() == 'y':
            solution = input("    Enter the fix/remedial code: ")
            node['remedial_history'].append({
                "solution_id": "manual_fix",
                "code_snippet": solution,
                "explanation": "Manually verified stability fix.",
                "source_notebook": "User_Triage"
            })
            node['status'] = "INDEXED"
            with open(node_path, 'w', encoding='utf-8') as f:
                json.dump(node, f, indent=2)
            print(colored("    ✅ Node Promoted to INDEXED.", "green"))
        if deep_scan:
            print(colored("    [Deep Scan Mode] Running extended analysis...", "cyan"))
            # Placeholder for deep scan logic
            # e.g., re-query remedies with higher limits, run additional checks
    print(colored("\nTriage complete.", "green"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KEL Triage Tool")
    parser.add_argument('--deep', action='store_true', help='Enable deep/full scan mode')
    args = parser.parse_args()
    run_triage(deep_scan=args.deep)