"""Dataset loaders for multi-agent benchmarking"""

import networkx as nx
import numpy as np
import os
import tarfile
import urllib.request
from typing import Tuple, Dict, Any
from sklearn.datasets import make_classification, make_moons
from sklearn.preprocessing import StandardScaler


def available_datasets():
    """Return list of available datasets"""
    return ['karate', 'cora', 'power_grid', 'synthetic_classification', 'synthetic_moons']


def load_karate_club() -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load Karate Club network
    Returns: (graph, features, labels)
    """
    G = nx.karate_club_graph()
    
    # Create features: degree, clustering coefficient, betweenness
    features = []
    for node in G.nodes():
        degree = G.degree(node)
        clustering = nx.clustering(G, node)
        features.append([degree, clustering])
    
    features = np.array(features)
    
    # Labels: Mr. Hi (0) vs Officer (1)
    labels = np.array([G.nodes[i]['club'] == 'Officer' for i in G.nodes()]).astype(int)
    
    return G, features, labels


def _download_cora_dataset(data_dir: str = 'data') -> str:
    """
    Download Cora dataset from LINQS website if not already present
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        Path to the extracted dataset directory
    """
    os.makedirs(data_dir, exist_ok=True)
    cora_dir = os.path.join(data_dir, 'cora')
    
    # Check if already downloaded
    content_file = os.path.join(cora_dir, 'cora.content')
    cites_file = os.path.join(cora_dir, 'cora.cites')
    
    if os.path.exists(content_file) and os.path.exists(cites_file):
        return cora_dir
    
    # Download the dataset
    url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    tgz_path = os.path.join(data_dir, 'cora.tgz')
    
    print(f"Downloading Cora dataset from {url}...")
    print("This may take a few minutes...")
    
    try:
        urllib.request.urlretrieve(url, tgz_path)
        print("Download complete. Extracting...")
        
        # Extract the tar.gz file
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        # Remove the tgz file
        os.remove(tgz_path)
        print("Extraction complete.")
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Cora dataset: {e}\n"
            f"Please manually download from {url} and extract to {cora_dir}"
        )
    
    return cora_dir


def load_cora(data_dir: str = 'data') -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load full Cora citation network dataset
    
    The Cora dataset consists of:
    - 2,708 papers (nodes)
    - 5,429 citations (edges)
    - 1,433 binary word features per paper
    - 7 classes (paper topics)
    
    Args:
        data_dir: Directory where dataset is stored (will download if not present)
        
    Returns:
        (graph, features, labels)
        - graph: NetworkX directed graph with paper citations
        - features: numpy array of shape (n_nodes, 1433) with binary word features
        - labels: numpy array of shape (n_nodes,) with class labels (0-6)
    """
    # Download dataset if needed
    cora_dir = _download_cora_dataset(data_dir)
    
    content_file = os.path.join(cora_dir, 'cora.content')
    cites_file = os.path.join(cora_dir, 'cora.cites')
    
    if not os.path.exists(content_file) or not os.path.exists(cites_file):
        raise FileNotFoundError(
            f"Cora dataset files not found. Expected:\n"
            f"  - {content_file}\n"
            f"  - {cites_file}\n"
            f"Please download from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
        )
    
    # Parse content file: paper_id feature1 feature2 ... feature1433 class_label
    paper_to_index = {}
    features_list = []
    labels_list = []
    
    with open(content_file, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            paper_id = parts[0]
            paper_to_index[paper_id] = idx
            
            # Features are binary (0 or 1)
            features = np.array([int(x) for x in parts[1:-1]], dtype=np.float32)
            features_list.append(features)
            
            # Label is the last element
            label = parts[-1]
            labels_list.append(label)
    
    # Convert to numpy arrays
    features = np.array(features_list)
    n_nodes = len(features)
    n_features = features.shape[1]  # Should be 1433
    
    # Map string labels to integers
    unique_labels = sorted(set(labels_list))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_to_int[label] for label in labels_list], dtype=int)
    n_classes = len(unique_labels)  # Should be 7
    
    # Build citation graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    
    with open(cites_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                cited_paper = parts[0]
                citing_paper = parts[1]
                
                # Only add edges for papers that exist in content file
                if cited_paper in paper_to_index and citing_paper in paper_to_index:
                    cited_idx = paper_to_index[cited_paper]
                    citing_idx = paper_to_index[citing_paper]
                    # Citation: citing_paper -> cited_paper (citing paper cites cited paper)
                    G.add_edge(citing_idx, cited_idx)
    
    # Ensure all nodes are in the graph (some papers might not have citations)
    for idx in range(n_nodes):
        if idx not in G:
            G.add_node(idx)
    
    # Convert to integer node labels if needed
    if not all(isinstance(n, int) for n in G.nodes()):
        G = nx.convert_node_labels_to_integers(G)
    
    print(f"Loaded Cora dataset: {n_nodes} papers, {G.number_of_edges()} citations, "
          f"{n_features} features, {n_classes} classes")
    
    return G, features, labels


def load_power_grid() -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load US Power Grid network
    Returns: (graph, features, labels)
    """
    try:
        # Try to load the actual power grid network
        G = nx.Graph()
        # Create a simplified power grid-like network
        # Real data would require external download
        n_nodes = 150
        G = nx.watts_strogatz_graph(n_nodes, 4, 0.3, seed=42)
    except:
        # Fallback to synthetic
        n_nodes = 150
        G = nx.watts_strogatz_graph(n_nodes, 4, 0.3, seed=42)
    
    # Create features based on graph properties
    features = []
    for node in G.nodes():
        degree = G.degree(node)
        clustering = nx.clustering(G, node)
        try:
            betweenness = nx.betweenness_centrality(G)[node]
        except:
            betweenness = 0.0
        features.append([degree, clustering, betweenness])
    
    features = np.array(features)
    
    # Create synthetic labels (e.g., node type/region)
    labels = np.random.RandomState(42).randint(0, 3, size=len(G.nodes()))
    
    return G, features, labels


def load_synthetic_classification(n_nodes: int = 100, n_features: int = 10, 
                                  n_classes: int = 2, n_samples: int = None, **kwargs) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Generate synthetic classification dataset with random graph
    
    Args:
        n_nodes: Number of nodes in the graph
        n_features: Number of features per sample
        n_classes: Number of classes
        n_samples: Total number of samples (can be > n_nodes for more data per node)
        **kwargs: Additional parameters (edge_probability, etc.)
    
    Returns: (graph, features, labels)
    """
    # If n_samples not specified, use n_nodes
    if n_samples is None:
        n_samples = n_nodes
    
    # Generate classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(1, n_features // 4),
        n_classes=n_classes,
        random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create random graph
    p = kwargs.get('edge_probability', 0.15)
    G = nx.erdos_renyi_graph(n_nodes, p, seed=42)
    
    # Ensure graph is connected
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)
        # Adjust data if graph was reduced
        if len(G.nodes()) < n_nodes:
            X = X[:len(G.nodes())]
            y = y[:len(G.nodes())]
    
    return G, X, y


def load_synthetic_moons(n_nodes: int = 100, n_samples: int = None, **kwargs) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Generate synthetic moons dataset with random graph
    
    Args:
        n_nodes: Number of nodes in the graph
        n_samples: Total number of samples (can be > n_nodes for more data per node)
        **kwargs: Additional parameters (n_features, edge_probability, etc.)
    
    Returns: (graph, features, labels)
    """
    # If n_samples not specified, use n_nodes
    if n_samples is None:
        n_samples = n_nodes
    
    # Generate moons data
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    
    # Add more features
    n_features = kwargs.get('n_features', 5)
    if n_features > 2:
        extra_features = np.random.RandomState(42).randn(n_samples, n_features - 2)
        X = np.hstack([X, extra_features])
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create random graph
    p = kwargs.get('edge_probability', 0.15)
    G = nx.erdos_renyi_graph(n_nodes, p, seed=42)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)
        # Adjust data if graph was reduced
        if len(G.nodes()) < n_nodes:
            X = X[:len(G.nodes())]
            y = y[:len(G.nodes())]
    
    return G, X, y


def load_dataset(name: str, **kwargs) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Load dataset by name
    
    Args:
        name: Dataset name ('karate', 'cora', 'power_grid', 'synthetic_classification', 'synthetic_moons')
        **kwargs: Additional arguments for synthetic datasets
        
    Returns:
        (graph, features, labels)
    """
    loaders = {
        'karate': load_karate_club,
        'cora': load_cora,
        'power_grid': load_power_grid,
        'synthetic_classification': load_synthetic_classification,
        'synthetic_moons': load_synthetic_moons
    }
    
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    
    if name.startswith('synthetic'):
        return loaders[name](**kwargs)
    else:
        return loaders[name]()

