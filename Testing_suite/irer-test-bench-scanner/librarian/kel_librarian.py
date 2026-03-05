import os
import json
import hashlib
import re
import uuid
import shutil
import traceback
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions


# --- CONFIGURATION (Centralized) ---
try:
    from core.librarian.kel_settings import Configuration, Query
    DIRS = Configuration.DIRS
    FAILURE_KEYWORDS = Query.FAILURE_KEYWORDS
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KEL_STORE_ROOT = os.path.join(BASE_DIR, ".kel_store")
    STAGING_ROOT = os.path.join(BASE_DIR, "staging")
    DIRS = {
        "NODES": os.path.join(KEL_STORE_ROOT, "nodes"),
        "CHROMA": os.path.join(KEL_STORE_ROOT, "chroma_db"),
        "INCOMING": os.path.join(STAGING_ROOT, "incoming"),
        "PROCESSED": os.path.join(STAGING_ROOT, "processed"),
        "FAILED": os.path.join(STAGING_ROOT, "failed")
    }
    FAILURE_KEYWORDS = ["FAILED", "ERROR", "EXCEPTION", "EXIT CODE", "nan", "inf", "divergence", "collapse", "unstable"]

# Ensure directories exist
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# --- SCHEMA (Same as before) ---
class RemedialAction(BaseModel):
    solution_id: str
    code_snippet: str
    explanation: str
    effectiveness_score: float = 0.8
    source_notebook: str

class KnowledgeNode(BaseModel):
    uid: str
    status: str = "INDEXED" 
    kind: str = "FRICTION_POINT"
    semantic_data: Dict[str, Any] = Field(default_factory=dict)
    pattern_signature: Dict[str, str] = Field(default_factory=dict)
    remedial_history: List[RemedialAction] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

from core.librarian.kel_settings import Telemetry
import logging

# --- THE LIBRARIAN ---
class KelLibrarian:
    def __init__(self):
        log_level = getattr(logging, Telemetry.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')
        logging.info("📚 [KEL] Initializing Librarian...")
        logging.info(f"   📂 Watch Folder: {DIRS['INCOMING']}")
        self.client = chromadb.PersistentClient(path=DIRS['CHROMA'])
        self.collection = self.client.get_or_create_collection(
            name="remedial_manifold",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )

    def _generate_uid(self, content: str) -> str:
        return hashlib.md5(content.strip().encode('utf-8')).hexdigest()

    def _parse_bundle_file(self, file_path: str) -> List[Dict]:
        notebooks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        sections = re.split(r'#{80}\nFILE:', content)
        for section in sections[1:]:
            try:
                lines = section.split('\n')
                filename = lines[0].strip()
                json_start = section.find('{')
                if json_start == -1: continue
                notebook_data = json.loads(section[json_start:])
                notebooks.append({"filename": filename, "data": notebook_data})
            except Exception as e:
                logging.warning(f"⚠️ Parsing Error in bundle section: {e}")
        return notebooks

    def run_ingestion_cycle(self):
        """Scans 'incoming', processes files in parallel, and moves them. Uses RAM for batch processing. Adds progress bar and logging."""
        import concurrent.futures
        from tqdm import tqdm
        logging.info("🔄 [KEL] Starting Ingestion Cycle...")
        files = [f for f in os.listdir(DIRS['INCOMING']) if os.path.isfile(os.path.join(DIRS['INCOMING'], f))]
        if not files:
            logging.info("💤 No files found in 'incoming'.")
            return

        import time
        def process_file(file_name):
            src_path = os.path.join(DIRS['INCOMING'], file_name)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # 1. Process
                    if file_name.endswith(".txt") and "Category_" in file_name:
                        notebooks = self._parse_bundle_file(src_path)
                        for nb in notebooks:
                            self._analyze_notebook(nb['data'], nb['filename'])
                    elif file_name.endswith(".ipynb"):
                        with open(src_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self._analyze_notebook(data, file_name)
                    # 2. Move to Processed
                    dest_path = os.path.join(DIRS['PROCESSED'], file_name)
                    shutil.move(src_path, dest_path)
                    logging.info(f"✅ Success: Moved to 'processed/' {file_name}")
                    break
                except Exception as e:
                    logging.error(f"❌ FAILED: {file_name} Attempt {attempt+1}/{max_retries} {e}")
                    traceback.print_exc()
                    time.sleep(1)
                    if attempt == max_retries - 1:
                        # 3. Move to Failed
                        dest_path = os.path.join(DIRS['FAILED'], file_name)
                        if os.path.exists(dest_path): os.remove(dest_path)
                        try:
                            shutil.move(src_path, dest_path)
                        except Exception as move_exc:
                            logging.error(f"❌ FAILED to move {file_name} to failed: {move_exc}")
                            traceback.print_exc()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(process_file, files), total=len(files), desc="Ingesting Files"))

    def _analyze_notebook(self, nb_data: Dict, source_name: str):
        # Validate and clean notebook data
        if not isinstance(nb_data, dict) or 'cells' not in nb_data:
            logging.warning(f"Notebook {source_name} is invalid or missing cells.")
            return
        cells = nb_data.get('cells', [])
        context_buffer = []
        try:
            from core.librarian.kel_settings import Query
            FAILURE_KEYWORDS = [k.lower() for k in Query.FAILURE_KEYWORDS]
            LOOKAHEAD_DEPTH = Query.LOOKAHEAD_DEPTH
            SEARCH_LIMIT = Query.SEARCH_LIMIT
            SIMILARITY_THRESHOLD = Query.SIMILARITY_THRESHOLD
        except Exception:
            FAILURE_KEYWORDS = [k.lower() for k in ["FAILED", "ERROR", "EXCEPTION", "EXIT CODE", "nan", "inf", "divergence", "collapse", "unstable"]]
            LOOKAHEAD_DEPTH = 5
            SEARCH_LIMIT = 3
            SIMILARITY_THRESHOLD = 0.85

        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'markdown':
                text = "".join(cell.get('source', []))
                context_buffer.append(text)
                if len(context_buffer) > 2:
                    context_buffer.pop(0)

            elif cell.get('cell_type') == 'code':
                outputs = cell.get('outputs', [])
                error = None
                matched_keyword = None
                for o in outputs:
                    # Flexible keyword matching in error/stream outputs
                    if o.get('output_type') == 'error':
                        error = o
                        matched_keyword = 'error'
                        break
                    if o.get('output_type') == 'stream':
                        stream_text = "".join(o.get('text', [])).lower()
                        for k in FAILURE_KEYWORDS:
                            if k in stream_text:
                                error = o
                                matched_keyword = k
                                break
                        if error:
                            break

                if error:
                    # Integrate vector similarity search for error context
                    try:
                        from core.librarian.kel_interface import KelInterface
                        kel = KelInterface()
                        remedies = kel.query_remedies(
                            error_trace=matched_keyword if matched_keyword else 'error',
                            context="\n".join(context_buffer),
                            limit=SEARCH_LIMIT
                        )
                        remedies = [r for r in remedies if r.get('similarity_score', 1.0) <= SIMILARITY_THRESHOLD]
                    except Exception:
                        remedies = []

                    # Extract variable values and execution environment if available
                    variable_values = {}
                    env_info = {}
                    if 'metadata' in cell:
                        env_info = cell['metadata']
                    # Full traceback
                    full_traceback = error.get('traceback', []) if error.get('output_type') == 'error' else error.get('text', [])

                    # Tagging: severity, reproducibility, frequency
                    severity = 'high' if matched_keyword in ['error', 'failed', 'exception'] else 'medium'
                    reproducibility = 'unknown'
                    frequency = 1

                    self._create_friction_point(
                        error_trace=full_traceback,
                        context=context_buffer,
                        code_snippet="".join(cell.get('source', [])),
                        source=source_name,
                        next_cells=cells[i+1:i+1+LOOKAHEAD_DEPTH],
                        remedies=remedies,
                        cell_meta={
                            'variable_values': variable_values,
                            'execution_environment': env_info,
                            'severity': severity,
                            'reproducibility': reproducibility,
                            'frequency': frequency
                        }
                    )

    def _create_friction_point(self, error_trace, context, code_snippet, source, next_cells, remedies=None, cell_meta=None):
        trace_str = "\n".join(error_trace) if isinstance(error_trace, list) else str(error_trace)
        description = "\n".join(context) if context else "No context provided."
        uid = self._generate_uid(trace_str + code_snippet)

        solutions = []
        for cell in next_cells:
            if cell['cell_type'] == 'code' and cell.get('outputs') and not any(o.get('output_type') == 'error' for o in cell.get('outputs')):
                 solutions.append(RemedialAction(
                    solution_id=str(uuid.uuid4())[:8],
                    code_snippet="".join(cell.get('source', [])),
                    explanation="Auto-extracted subsequent successful execution.",
                    source_notebook=source
                 ))
                 break

        # Store detailed context: cell content, output, and metadata
        meta = {
            "source": source,
            "ingested_at": datetime.now().isoformat(),
            "remedies": remedies if remedies else [],
            "cell_content": code_snippet,
            "cell_output": error_trace,
            "cell_meta": cell_meta if cell_meta else {}
        }

        node = KnowledgeNode(
            uid=uid,
            status="INDEXED" if solutions else "UNINDEXED",
            semantic_data={"description": description, "tags": ["auto-ingest"], "vector_id": uid},
            pattern_signature={"error_trace_fingerprint": trace_str[:500]},
            remedial_history=solutions,
            meta=meta
        )

        with open(os.path.join(DIRS['NODES'], f"{uid}.json"), 'w') as f:
            f.write(node.model_dump_json(indent=2))

        embed_text = f"CONTEXT: {description}\nERROR: {trace_str}\nCODE: {code_snippet}"
        self.collection.upsert(
            documents=[embed_text],
            metadatas=[{"uid": uid, "source": source, "has_fix": bool(solutions)}],
            ids=[uid]
        )
        logging.info(f"Found Friction Point: {uid[:8]} (Fix: {len(solutions)})")

    def export_results(self, format="json", output_path=None):
        """
        Export all KnowledgeNodes in NODES to JSON, CSV, or Markdown.
        """
        import pandas as pd
        nodes = []
        for filename in os.listdir(DIRS['NODES']):
            if filename.endswith('.json'):
                with open(os.path.join(DIRS['NODES'], filename), 'r', encoding='utf-8') as f:
                    node = json.load(f)
                    nodes.append(node)
        df = pd.DataFrame(nodes)
        if format == "json":
            result = df.to_json(orient="records", indent=2)
        elif format == "csv":
            result = df.to_csv(index=False)
        elif format == "md" or format == "markdown":
            result = df.to_markdown(index=False)
        else:
            raise ValueError("Unsupported export format")
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        return result

    # Unit test outline
    def test_friction_point_creation(self):
        """Unit test for friction point creation."""
        dummy_trace = ["Dummy error"]
        dummy_context = ["Dummy context"]
        dummy_code = "print('hello')"
        dummy_source = "dummy.ipynb"
        dummy_cells = []
        self._create_friction_point(dummy_trace, dummy_context, dummy_code, dummy_source, dummy_cells)
        logging.info("Unit test: Friction point creation completed.")

    # Visualization outline
    def visualize_friction_points(self):
        """
        Optionally generate charts/graphs for friction points using matplotlib.
        """
        import matplotlib.pyplot as plt
        nodes = []
        for filename in os.listdir(DIRS['NODES']):
            if filename.endswith('.json'):
                with open(os.path.join(DIRS['NODES'], filename), 'r', encoding='utf-8') as f:
                    node = json.load(f)
                    nodes.append(node)
        statuses = [n.get('status', 'UNKNOWN') for n in nodes]
        plt.figure(figsize=(6,4))
        plt.hist(statuses, bins=len(set(statuses)), color='skyblue', edgecolor='black')
        plt.title('Friction Point Status Distribution')
        plt.xlabel('Status')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    # Parallel processing entry point (outline)
    def batch_analyze_notebooks(self, notebook_list):
        import concurrent.futures
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_nb = {executor.submit(self._analyze_notebook, nb['data'], nb['filename']): nb for nb in notebook_list}
            for future in concurrent.futures.as_completed(future_to_nb):
                nb = future_to_nb[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f"Notebook {nb['filename']} generated an exception: {exc}")
        return results

if __name__ == "__main__":
    librarian = KelLibrarian()
    librarian.run_ingestion_cycle()