
import numpy as np
import sys
from sklearn.cluster import DBSCAN  # type: ignore[import]

try:
    from ripser import ripser  # type: ignore[import]
except ImportError:
    print("FATAL: 'ripser' is required for TDA. Run: pip install ripser", file=sys.stderr)
    sys.exit(1)

def extract_and_classify_topology(rho_field: np.ndarray, persistence_threshold: float = 0.5, seed: int = 42):
    """
    Executes the TDA Quantule Census on a 3D resonance density field.
    Returns the formatted CSV string and a dictionary of taxonomy counts.
    """
    np.random.seed(seed)
    # 1. Quantule Census Filter: mu + 3*sigma
    mu = np.mean(rho_field)
    sigma = np.std(rho_field)
    critical_threshold = mu + (3 * sigma)
    
    # Extract the event cloud
    coords = np.argwhere(rho_field > critical_threshold)

    # Mathematical Hardening: Prevent Ripser OOM Hangs
    MAX_TDA_POINTS = 1000
    if len(coords) > MAX_TDA_POINTS:
        rng = np.random.default_rng(seed)  # <-- HOTFIX: Locked PRNG
        indices = rng.choice(len(coords), size=MAX_TDA_POINTS, replace=False)
        coords = coords[indices]

    csv_header = "quantule_id,type,center_x,center_y,center_z,radius,magnitude"
    csv_lines = [csv_header]
    taxonomy_counts = {"Q_theta": 0, "Q_nu": 0, "Transient": 0}

    # If the field is totally flat/vacuum, return empty
    if len(coords) < 5:
        return "\n".join(csv_lines) + "\n", taxonomy_counts

    # 2. DBSCAN Clustering (Isolating individual Quantules)
    # eps=2.0 connects adjacent diagonal cells in the 3D grid
    clustering = DBSCAN(eps=2.0, min_samples=4).fit(coords)
    labels = clustering.labels_
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue # Skip unclustered noise

        cluster_coords = coords[labels == label]
        if len(cluster_coords) < 5:
            continue

        # Calculate geometric metadata
        center = np.mean(cluster_coords, axis=0)
        radius = np.max(np.linalg.norm(cluster_coords - center, axis=1))
        
        # Get the max resonance density (magnitude) of this specific quantule
        magnitudes = [rho_field[tuple(c)] for c in cluster_coords]
        magnitude = np.max(magnitudes)

        # 3. Persistent Homology (Ripser)
        # Mean-center the cluster coordinates to ensure numerical stability for Ripser
        centered_coords = cluster_coords - center
        
        try:
            # maxdim=2 calculates H0, H1 (loops), and H2 (voids)
            dgms = ripser(centered_coords, maxdim=2)['dgms']
            H1 = dgms[1] if len(dgms) > 1 else []
            H2 = dgms[2] if len(dgms) > 2 else []

            # 4. Persistence Filtering (> 0.5 lifespan)
            persistent_loops = sum(1 for d in H1 if (d[1] - d[0]) > persistence_threshold)
            persistent_voids = sum(1 for d in H2 if (d[1] - d[0]) > persistence_threshold)

            # 5. Taxonomy Classification
            if persistent_voids > 0:
                q_type = "Q_theta" # Synchronizers (H2 voids)
            elif persistent_loops > 0:
                q_type = "Q_nu"    # Entropy Sinks (H1 loops)
            else:
                q_type = "Transient" # Unstable collapse
                
        except Exception as e:
            print(f"Warning: Ripser failed on cluster {label}: {e}")
            q_type = "Transient"

        taxonomy_counts[q_type] += 1
        
        # Append to CSV ledger
        csv_lines.append(f"q{label},{q_type},{center[0]:.2f},{center[1]:.2f},{center[2]:.2f},{radius:.2f},{magnitude:.6f}")

    return "\n".join(csv_lines) + "\n", taxonomy_counts