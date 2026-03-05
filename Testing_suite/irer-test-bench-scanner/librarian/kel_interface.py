import os
import json
import chromadb
from typing import List, Dict, Optional
from pydantic import BaseModel

# --- CONFIGURATION (Centralized) ---
try:
    from core.librarian.kel_settings import Configuration, Query
    KEL_STORE_ROOT = Configuration.KEL_STORE_ROOT
    NODES_DIR = Configuration.DIRS["NODES"]
    CHROMA_DIR = Configuration.DIRS["CHROMA"]
    FAILURE_KEYWORDS = Query.FAILURE_KEYWORDS
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KEL_STORE_ROOT = os.path.join(BASE_DIR, ".kel_store")
    NODES_DIR = os.path.join(KEL_STORE_ROOT, "nodes")
    CHROMA_DIR = os.path.join(KEL_STORE_ROOT, "chroma_db")
    FAILURE_KEYWORDS = ["FAILED", "ERROR", "EXCEPTION", "EXIT CODE", "nan", "inf", "divergence", "collapse", "unstable"]

class KelInterface:
    def __init__(self):
        """Initializes connection to the Knowledge Extraction Log."""
        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError(f"KEL Store not found at {KEL_STORE_ROOT}. Run kel_librarian.py first.")
            
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_collection(name="remedial_manifold")
        print(f"🔌 [INTERFACE] Connected to KEL Remedial Manifold.")

    def query_remedies(self, error_trace: str, context: str = "", limit: int = 10, similarity_threshold: float = 0.95) -> List[Dict]:
        """
        Semantic Search: Finds historical fixes for a given error trace.
        Returns remedies with recurrence_count and resolution_rate for UI.
        """
        query_text = f"CONTEXT: {context}\nERROR: {error_trace}"
        results = self.collection.query(
            query_texts=[query_text],
            n_results=limit,
        )
        remedies = []
        if results['ids']:
            for i, uid in enumerate(results['ids'][0]):
                node = self.get_node(uid)
                if node:
                    # 1. Calculate Resolution Rate (Average effectiveness of fixes)
                    fixes = node.get("remedial_history", [])
                    resolution_rate = 0.0
                    if fixes:
                        resolution_rate = sum(f.get('effectiveness_score', 0.8) for f in fixes) / len(fixes)
                    # 2. Get Recurrence Count
                    recurrence = node.get("meta", {}).get("frequency", node.get("meta", {}).get("occurrences", 1))
                    # 3. Convert similarity (distance) to similarity_score
                    dist = results['distances'][0][i] if 'distances' in results else 0
                    similarity_score = 1.0 - dist
                    remedy = {
                        "uid": uid,
                        "similarity_score": similarity_score,
                        "description": node.get("semantic_data", {}).get("description", ""),
                        "code_snippet": fixes[0].get("code_snippet", "") if fixes else "",
                        "fixes": fixes,
                        "recurrence_count": recurrence,
                        "resolution_rate": resolution_rate
                    }
                    if similarity_score >= similarity_threshold:
                        remedies.append(remedy)
        # Rank remedies by resolution_rate, then similarity_score
        remedies.sort(key=lambda r: (-r["resolution_rate"], -r["similarity_score"]))
        return remedies
    def get_active_friction_nodes(self) -> List[Dict]:
        """Returns UNINDEXED and URGENT nodes for the UI's UnifiedLensExplorer."""
        active_nodes = []
        if not os.path.exists(NODES_DIR):
            return active_nodes
        for filename in os.listdir(NODES_DIR):
            if filename.endswith(".json"):
                with open(os.path.join(NODES_DIR, filename), 'r', encoding='utf-8') as f:
                    node = json.load(f)
                    if node.get("status") in ["UNINDEXED", "URGENT"]:
                        active_nodes.append({
                            "id": node.get("uid"),
                            "file": node.get("meta", {}).get("source", "Unknown"),
                            "tier": "red" if node.get("status") == "URGENT" else "amber",
                            "description": node.get("semantic_data", {}).get("description", "")
                        })
        return active_nodes

    def batch_query_remedies(self, risk_zones: List[Dict], limit: int = 10, similarity_threshold: float = 0.95) -> List[Dict]:
        """
        Batch query for multiple risk zones.
        Returns aggregated remedies for all zones.
        """
        all_remedies = []
        for zone in risk_zones:
            context = f"Using {zone['name']} in function {zone.get('context', 'global_scope')}"
            error_trace = f"Potential instability in {zone['name']} configuration."
            remedies = self.query_remedies(error_trace, context, limit=limit, similarity_threshold=similarity_threshold)
            all_remedies.extend(remedies)
        # Optionally deduplicate remedies by uid
        seen = set()
        deduped = []
        for r in all_remedies:
            if r["uid"] not in seen:
                deduped.append(r)
                seen.add(r["uid"])
        return deduped

    def get_node(self, uid: str) -> Optional[Dict]:
        """Retrieves the full JSON Knowledge Node."""
        path = os.path.join(NODES_DIR, f"{uid}.json")
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load node {uid}: {e}")
        return None

    def get_unindexed_nodes(self) -> List[Dict]:
        """Returns all nodes marked as UNINDEXED (No solution found)."""
        unindexed = []
        try:
            for filename in os.listdir(NODES_DIR):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(NODES_DIR, filename), 'r', encoding='utf-8') as f:
                            node = json.load(f)
                            if node.get("status") == "UNINDEXED":
                                unindexed.append(node)
                    except Exception as e:
                        print(f"[ERROR] Failed to load node file {filename}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to list nodes directory: {e}")
        return unindexed

    def stats(self):
        """Returns database health metrics."""
        count = self.collection.count()
        nodes = len(os.listdir(NODES_DIR))
        return {"vector_entries": count, "json_nodes": nodes}

# --- CLI TESTING UTILITY ---
if __name__ == "__main__":
    interface = KelInterface()
    
    print("\n--- 📊 SYSTEM STATS ---")
    print(interface.stats())
    
    print("\n--- 🧪 TEST QUERY: 'Vacuum Collapse' ---")
    results = interface.query_remedies("g_tt collapsing to negative infinity", "Running simulation with alpha=0.9")
    for r in results:
        print(f"Found: {r['uid']} (Fixes: {len(r['fixes'])})")
        if r['fixes']:
            print(f"Example Fix Code: {r['fixes'][0]['code_snippet'][:50]}...")
            
    print("\n--- 🚩 UNINDEXED ISSUES ---")
    unindexed = interface.get_unindexed_nodes()
    print(f"Count: {len(unindexed)}")