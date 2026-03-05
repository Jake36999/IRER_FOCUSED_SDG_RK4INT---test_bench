import os
import json
from collections import defaultdict

# Root directory for IRER stack
ROOT = os.path.dirname(os.path.abspath(__file__))

MODULES = ["api", "data", "orchestrator", "redis", "validation", "worker"]

inventory = defaultdict(list)
metadata = defaultdict(dict)

for module in MODULES:
    module_path = os.path.join(ROOT, module)
    if os.path.isdir(module_path):
        for root, dirs, files in os.walk(module_path):
            rel_root = os.path.relpath(root, ROOT)
            for f in files:
                inventory[module].append(os.path.join(rel_root, f))
            # Metadata extraction
            for fname in ["README.md", "requirements.txt", "dockerfile", "Dockerfile"]:
                if fname in files:
                    metadata[module][fname] = os.path.join(rel_root, fname)
            for fname in files:
                if fname.endswith(".yaml") or fname.endswith(".json"):
                    metadata[module][fname] = os.path.join(rel_root, fname)
                if fname.startswith("test_") or "/tests/" in rel_root:
                    metadata[module]["tests"] = metadata[module].get("tests", []) + [os.path.join(rel_root, fname)]

# Root files
root_files = [f for f in os.listdir(ROOT) if os.path.isfile(os.path.join(ROOT, f))]

# Output
output = {
    "root_files": root_files,
    "inventory": dict(inventory),
    "metadata": dict(metadata)
}

with open(os.path.join(ROOT, "directory_inventory.json"), "w") as f:
    json.dump(output, f, indent=2)

print("Directory inventory and metadata extraction complete.")