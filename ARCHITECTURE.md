# Architecture Documentation

**Multi-Agent Network Topology Benchmarking Framework**

This document provides a comprehensive technical overview of the system architecture, data flow, algorithms, and data structures used throughout the framework.

---

## Table of Contents

1. [High-Level System Overview](#high-level-system-overview)
2. [Entry Point and Configuration](#entry-point-and-configuration)
3. [Dataset Loading](#dataset-loading)
4. [Network Topology Building](#network-topology-building)
5. [Agent System](#agent-system)
6. [Simulation Engine](#simulation-engine)
7. [Perturbations](#perturbations)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Visualization](#visualization)
10. [Reporting](#reporting)
11. [Detailed Data Flow](#detailed-data-flow)
12. [Dependencies and External Libraries](#dependencies-and-external-libraries)

---

## High-Level System Overview

### System Architecture

The framework follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Entry Point (cli.py)                    │
│                    - Loads YAML configuration                    │
│                    - Orchestrates experiments                   │
└────────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Dataset Loading Module                       │
│              (src/datasets/loaders.py)                          │
│         - Loads graph-structured datasets                      │
│         - Returns: (graph, features, labels)                   │
└────────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Network Topology Building                      │
│              (src/networks/topologies.py)                       │
│         - Builds communication network (9 topologies)          │
│         - Returns: NetworkX DiGraph with weights/delays        │
└────────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Creation                             │
│              (src/agents/models.py)                             │
│         - Creates N agents of specified model type              │
│         - Each agent has local model + aggregation strategy    │
└────────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Simulation Engine                            │
│              (src/simulation/engine.py)                         │
│   1. Distribute training data to agents                         │
│   2. Initial local training                                     │
│   3. Communication rounds loop:                                │
│      a) Apply perturbations                                     │
│      b) Communicate (message passing)                           │
│      c) Aggregate parameters                                    │
│      d) Evaluate system                                         │
└────────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation & Analysis                        │
│    - Metrics calculation (src/evaluation/metrics.py)           │
│    - Visualization (src/utils/visualization.py)                │
│    - Reporting (src/utils/reporting.py)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

1. **Configuration** → YAML file parsed into Python dict
2. **Dataset** → Graph, features array (N×F), labels array (N,)
3. **Topology** → NetworkX DiGraph with N nodes, weighted/delayed edges
4. **Agents** → List of N agent objects, each with local model
5. **Data Distribution** → Each agent receives subset of training data
6. **Simulation** → Rounds of communication, aggregation, evaluation
7. **Results** → Dictionary with metrics, history, network info
8. **Visualization** → Plots and reports generated

---

## Entry Point and Configuration

### File: `cli.py`

**Purpose**: Command-line interface that orchestrates the entire benchmarking process.

### Key Functions

#### `load_config(config_path: str) -> dict`

**Purpose**: Loads and parses YAML configuration file.

**Parameters**:
- `config_path`: Path to YAML configuration file

**Returns**: Python dictionary containing experiment configuration

**Algorithm**: Uses `yaml.safe_load()` to parse YAML into nested dictionary structure.

**Data Structure**:
```python
config = {
    'name': str,                    # Experiment name
    'dataset': str,                  # Dataset identifier
    'topology': str,                 # Topology type
    'n_agents': int,                 # Number of agents
    'agent_type': str,               # Model type (logistic, neural, gat, rf)
    'agent_params': dict,            # Model-specific hyperparameters
    'n_rounds': int,                 # Communication rounds
    'test_size': float,              # Test split ratio (0.0-1.0)
    'data_distribution': str,        # Distribution strategy
    'perturbations': list,           # List of perturbation configs
    'topology_params': dict          # Topology-specific parameters
}
```

#### `run_single_experiment(config: dict, verbose: bool = True) -> tuple`

**Purpose**: Executes a single experiment from configuration.

**Flow**:
1. Load dataset using `load_dataset()` from `src.datasets`
2. Split data into train/test using `train_test_split()` (70/30 default)
3. Build topology using `build_topology()` from `src.networks`
4. Create agents using appropriate agent class from `AGENT_TYPES` mapping
5. Initialize `SimulationEngine` with agents, network, test data
6. Distribute training data to agents
7. Add perturbations (if configured)
8. Run simulation for N rounds
9. Evaluate system and compute metrics
10. Return results dictionary and simulation engine

**Returns**: `(results: dict, sim: SimulationEngine)`

**Dependencies**:
- `src.datasets.load_dataset`
- `src.networks.build_topology`
- `src.agents` (various agent classes)
- `src.simulation.SimulationEngine`
- `src.evaluation.evaluate_system`

#### `run_benchmark(config_path: str, output_dir: str, verbose: bool = True) -> None`

**Purpose**: Runs multiple experiments from configuration file and generates comprehensive outputs.

**Flow**:
1. Load configuration (may contain multiple experiments)
2. For each experiment:
   - Run experiment via `run_single_experiment()`
   - Save results to CSV
   - Generate accuracy history plot
   - Generate network topology visualization
3. If multiple experiments:
   - Generate comparison plots
   - Generate summary plots
   - Generate comprehensive comparison (if ≥20 experiments)
   - Generate robustness/communication cost charts
4. Generate markdown report

**Output Files**:
- `{exp_name}_results.csv`: Flattened results
- `{exp_name}_accuracy.png`: Accuracy over time
- `{exp_name}_network.png`: Network visualization
- `comparison.png`: Multi-experiment comparison
- `benchmark_report.md`: Comprehensive report

**Dependencies**:
- `src.utils.save_results_to_csv`
- `src.utils.plot_accuracy_history`
- `src.utils.plot_network`
- `src.utils.create_comprehensive_comparison`
- `src.utils.plot_robustness_and_communication_cost`
- `src.utils.generate_report`

### Data Structures

**AGENT_TYPES Mapping**:
```python
AGENT_TYPES = {
    'logistic': LogisticAgent,
    'linear': LinearAgent,
    'neural': NeuralAgent,
    'gat': GATAgent,
    'gnn': GNNAgent,
    'graphsage': GraphSAGEAgent,
    'gin': GINAgent,
    'rf': RandomForestAgent,
    'random_forest': RandomForestAgent
}
```

---

## Dataset Loading

### File: `src/datasets/loaders.py`

**Purpose**: Loads graph-structured datasets for multi-agent learning experiments.

### Key Functions

#### `load_dataset(name: str, **kwargs) -> Tuple[nx.Graph, np.ndarray, np.ndarray]`

**Purpose**: Main dispatcher function that loads datasets by name.

**Parameters**:
- `name`: Dataset identifier ('cora', 'karate', 'synthetic_classification', etc.)
- `**kwargs`: Dataset-specific parameters

**Returns**: `(graph: nx.Graph, features: np.ndarray, labels: np.ndarray)`
- `graph`: NetworkX graph with N nodes
- `features`: Array of shape (N, F) where F is number of features
- `labels`: Array of shape (N,) with class labels

**Algorithm**: Dispatches to appropriate loader function based on name.

#### `load_cora(data_dir: str = 'data') -> Tuple[nx.Graph, np.ndarray, np.ndarray]`

**Purpose**: Loads Cora citation network dataset.

**Data Structure**:
- **Graph**: 2,708 papers (nodes), 5,429 citations (edges)
- **Features**: 1,433 binary word features per paper
- **Labels**: 7 classes (paper topics)

**Algorithm**:
1. Download dataset if not present (from LINQS website)
2. Parse `cora.content`: `paper_id feature1 ... feature1433 class_label`
3. Parse `cora.cites`: `cited_paper citing_paper` (creates directed edges)
4. Build NetworkX DiGraph with integer node IDs
5. Map string labels to integers (0-6)

**File Format**:
- `cora.content`: Tab-separated, one paper per line
- `cora.cites`: Tab-separated citation pairs

**Returns**: `(G: nx.DiGraph, features: np.ndarray[float32], labels: np.ndarray[int])`

#### `load_karate_club() -> Tuple[nx.Graph, np.ndarray, np.ndarray]`

**Purpose**: Loads Zachary's Karate Club network.

**Data Structure**:
- **Graph**: 34 nodes, 78 edges (undirected)
- **Features**: 2 features per node (degree, clustering coefficient)
- **Labels**: 2 classes (Mr. Hi vs Officer)

**Algorithm**:
1. Use NetworkX built-in `karate_club_graph()`
2. Compute node features: degree, clustering coefficient
3. Extract labels from node attribute 'club'

#### `load_synthetic_classification(n_nodes: int = 100, n_features: int = 10, n_classes: int = 2, **kwargs) -> Tuple[nx.Graph, np.ndarray, np.ndarray]`

**Purpose**: Generates synthetic classification dataset with random graph structure.

**Parameters**:
- `n_nodes`: Number of nodes in graph
- `n_features`: Number of features per sample
- `n_classes`: Number of classes
- `edge_probability`: Probability for Erdos-Renyi graph (default: 0.15)

**Algorithm**:
1. Generate classification data using `sklearn.make_classification()`
2. Standardize features using `StandardScaler`
3. Create random graph using `nx.erdos_renyi_graph()`
4. Ensure graph connectivity (extract largest connected component if needed)
5. Adjust data size if graph was reduced

**Returns**: `(G: nx.Graph, X: np.ndarray, y: np.ndarray)`

#### `load_synthetic_moons(n_nodes: int = 100, **kwargs) -> Tuple[nx.Graph, np.ndarray, np.ndarray]`

**Purpose**: Generates synthetic two-moons dataset with random graph.

**Algorithm**: Similar to `load_synthetic_classification()` but uses `make_moons()` for data generation.

### Data Structures

**Graph Structure**:
- Type: `networkx.Graph` or `networkx.DiGraph`
- Nodes: Integer IDs (0 to N-1)
- Edges: May have attributes (not used in dataset graph, but used in communication network)

**Features Array**:
- Shape: `(N, F)` where N = number of nodes, F = number of features
- Dtype: `np.float32` or `np.float64`
- Content: Feature vectors for each node

**Labels Array**:
- Shape: `(N,)`
- Dtype: `np.int32` or `np.int64`
- Content: Class labels (0 to C-1 where C = number of classes)

---

## Network Topology Building

### File: `src/networks/topologies.py`

**Purpose**: Constructs communication network topologies for multi-agent systems.

### Key Functions

#### `build_topology(topology_name: str, n_agents: int, **kwargs) -> nx.DiGraph`

**Purpose**: Main dispatcher that builds topology by name.

**Parameters**:
- `topology_name`: One of 9 topology types
- `n_agents`: Number of agents (nodes in graph)
- `**kwargs`: Topology-specific parameters

**Returns**: `networkx.DiGraph` with:
- Nodes: 0 to n_agents-1
- Edges: Directed edges with attributes:
  - `weight`: float (0.0-1.0), connection strength
  - `delay`: int (0+), communication delay in rounds

**Algorithm**: Dispatches to specific builder function based on topology_name.

#### Topology Builders

##### `build_star_topology(n_agents: int, **kwargs) -> nx.DiGraph`

**Purpose**: Creates star topology with central hub.

**Parameters**:
- `hub_id`: ID of hub node (default: 0)
- `bidirectional`: Whether edges are bidirectional (default: True)

**Algorithm**:
1. Create empty DiGraph
2. Add all nodes (0 to n_agents-1)
3. Connect all non-hub nodes to hub
4. If bidirectional, add reverse edges
5. Assign random weights (0.5-1.0) to each edge

**Structure**: Hub connects to all spokes, creating 2*(n_agents-1) edges if bidirectional.

##### `build_mesh_topology(n_agents: int, **kwargs) -> nx.DiGraph`

**Purpose**: Creates complete graph (all-to-all connections).

**Parameters**:
- `edge_probability`: Probability of edge (default: 1.0 for full mesh)

**Algorithm**:
1. Create empty DiGraph
2. Add all nodes
3. For each pair (i, j) where i ≠ j:
   - Add edge with random weight (0.5-1.0)
   - If bidirectional, add reverse edge

**Structure**: N*(N-1) edges (if bidirectional), maximum connectivity.

##### `build_ring_topology(n_agents: int, **kwargs) -> nx.DiGraph`

**Purpose**: Creates ring topology (circular chain).

**Algorithm**:
1. Create chain: 0→1→2→...→(N-1)
2. Close the ring: (N-1)→0
3. If bidirectional, add reverse edges
4. Assign random weights

**Structure**: N edges (unidirectional) or 2*N edges (bidirectional).

##### `build_cascade_topology(n_agents: int, **kwargs) -> nx.DiGraph`

**Purpose**: Creates sequential chain topology.

**Parameters**:
- `skip_connections`: Add skip connections (default: False)
- `bidirectional`: Add backward edges (default: False)

**Algorithm**:
1. Create chain: 0→1→2→...→(N-1)
2. Optionally add backward edges if bidirectional
3. Optionally add skip connections (i→i+2) with probability
4. Assign random weights and optional delays

**Structure**: N-1 forward edges, optionally N-1 backward edges, optionally skip edges.

##### `build_small_world_topology(n_agents: int, **kwargs) -> nx.DiGraph`

**Purpose**: Creates small-world network using Watts-Strogatz algorithm.

**Parameters**:
- `k`: Number of neighbors per node (default: 4)
- `rewire_prob`: Probability of rewiring (default: 0.3)

**Algorithm**:
1. Start with ring where each node connects to k/2 neighbors on each side
2. Rewire each edge with probability `rewire_prob` to random node
3. Convert to directed graph
4. Assign random weights

**Structure**: Combines local clustering with long-range connections.

##### `build_scale_free_topology(n_agents: int, **kwargs) -> nx.DiGraph`

**Purpose**: Creates scale-free network using Barabási-Albert algorithm.

**Parameters**:
- `m`: Number of edges to attach from new node (default: 2)

**Algorithm**:
1. Start with m nodes fully connected
2. Add nodes one by one, each connecting to m existing nodes
3. Connection probability proportional to existing node degree (preferential attachment)
4. Convert to directed graph
5. Assign random weights

**Structure**: Power-law degree distribution, few highly connected hubs.

##### `build_tree_topology(n_agents: int, **kwargs) -> nx.DiGraph`

**Purpose**: Creates tree topology (hierarchical structure).

**Parameters**:
- `branching_factor`: Children per node (default: 2)

**Algorithm**:
1. Build tree level by level
2. Root node (0) has branching_factor children
3. Each subsequent level has branching_factor children per parent
4. Continue until all N nodes are added
5. Assign random weights

**Structure**: N-1 edges, no cycles, hierarchical.

##### `build_grid_topology(n_agents: int, **kwargs) -> nx.DiGraph`

**Purpose**: Creates 2D grid topology.

**Parameters**:
- `rows`, `cols`: Grid dimensions (auto-calculated if not provided)

**Algorithm**:
1. Calculate grid dimensions (rows × cols ≈ n_agents)
2. Create 2D grid connections (up, down, left, right)
3. Optionally add diagonal connections
4. Convert to directed graph
5. Assign random weights

**Structure**: Regular grid with 4 or 8 neighbors per node.

##### `build_feedback_rewired_topology(n_agents: int, **kwargs) -> nx.DiGraph`

**Purpose**: Creates feedback-rewired topology from base topology.

**Parameters**:
- `base_topology`: Base topology type ('ring', 'mesh', 'random')
- `rewire_prob`: Probability of rewiring (default: 0.3)
- `feedback_prob`: Probability of adding feedback edge (default: 0.4)

**Algorithm**:
1. Build base topology
2. Rewire edges with probability `rewire_prob`
3. Add feedback edges (i→j where j < i) with probability `feedback_prob`
4. Assign random weights

**Structure**: Combines base structure with feedback loops.

### Data Structures

**NetworkX DiGraph**:
```python
G = nx.DiGraph()
G.add_node(i)  # Node ID: 0 to n_agents-1
G.add_edge(u, v, weight=0.8, delay=0)  # Edge with attributes
```

**Edge Attributes**:
- `weight`: float, connection strength (0.0-1.0), used in weighted aggregation
- `delay`: int, communication delay in rounds, messages queued until delay expires

**Graph Properties**:
- Directed: Edges have direction (u→v means u sends to v)
- Weighted: Edges have weight attribute
- Delayed: Edges can have delay attribute for modeling latency

---

## Agent System

### Base Agent Class

### File: `src/agents/base_agent.py`

**Purpose**: Abstract base class defining interface for all agent types.

#### Class: `Agent(ABC)`

**Data Attributes**:
```python
agent_id: int                    # Unique identifier (0 to n_agents-1)
n_features: int                  # Number of input features
n_classes: int                   # Number of output classes
model: Any                       # The actual ML model (varies by subclass)
failed: bool                     # Failure state flag
message_queue: List[Dict]        # Queue of messages with delays
received_messages: List[Dict]    # History of processed messages
local_data: np.ndarray           # Local training features
local_labels: np.ndarray         # Local training labels
local_test_data: np.ndarray      # Local test features
local_test_labels: np.ndarray    # Local test labels
local_test_indices: np.ndarray  # Original test indices
prediction_history: List         # History of predictions
```

**Abstract Methods**:
- `train(X, y)`: Train local model on data
- `predict(X)`: Make predictions on data
- `get_model_params()`: Extract parameters for sharing
- `update_from_messages(messages)`: Update model from received parameters

**Concrete Methods**:
- `receive_message(message, delay)`: Add message to queue with delay
- `process_messages(current_time)`: Process ready messages (delay expired)
- `set_local_data(X, y)`: Set training data
- `set_local_test_data(X, y, indices)`: Set test data
- `fail()`: Mark agent as failed
- `recover()`: Mark agent as recovered
- `reset()`: Reset agent state

**Message Queue Structure**:
```python
message_queue = [
    {
        'message': {
            'params': {...},      # Model parameters
            'weight': float,      # Edge weight
            'sender': int,        # Sender agent ID
            'receiver': int       # Receiver agent ID
        },
        'delay': int,            # Rounds until message is ready
        'timestamp': int         # Time when message was received
    },
    ...
]
```

**Algorithm - `process_messages(current_time)`**:
1. Iterate through message_queue
2. If `current_time >= timestamp + delay`, message is ready
3. Collect ready messages
4. Update queue (remove ready messages)
5. If agent not failed, call `update_from_messages(ready_messages)`

### Agent Model Implementations

### File: `src/agents/models.py`

**Purpose**: Concrete agent implementations with different ML models.

#### Class: `LogisticAgent(Agent)`

**Model**: scikit-learn `LogisticRegression`

**Parameters**:
- `C`: Regularization strength (default: 1.0)
- `max_iter`: Maximum iterations (default: 1000)
- `aggregation_strategy`: Aggregation strategy instance

**Data Structures**:
```python
model.coef_: np.ndarray          # Shape: (n_classes, n_features) or (n_features,)
model.intercept_: np.ndarray     # Shape: (n_classes,) or scalar
```

**Methods**:
- `train(X, y)`: Calls `model.fit(X, y)`
- `predict(X)`: Calls `model.predict(X)`
- `get_model_params()`: Returns `{'coef': coef_, 'intercept': intercept_, 'trained': bool}`
- `update_from_messages(messages)`: Uses aggregation strategy to combine parameters

#### Class: `NeuralAgent(Agent)`

**Model**: PyTorch `SimpleNN` (multi-layer perceptron)

**Architecture**:
- Input layer: n_features → hidden_dim
- Hidden layers: hidden_dim → hidden_dim (n_layers-2 times)
- Output layer: hidden_dim → n_classes
- Batch normalization after each hidden layer
- Dropout for regularization
- ReLU activations

**Parameters**:
- `hidden_dim`: Hidden layer size (default: 64)
- `n_layers`: Number of layers (default: 3, max: 5)
- `lr`: Learning rate (default: 0.001)
- `epochs`: Training epochs (default: 100)
- `dropout`: Dropout probability (default: 0.2)

**Data Structures**:
```python
model.state_dict(): Dict[str, torch.Tensor]  # All layer weights and biases
# Keys: 'layers.0.weight', 'layers.0.bias', 'batch_norms.0.weight', etc.
```

**Methods**:
- `train(X, y)`: 
  - Normalize features (mean/std scaling)
  - Train with Adam optimizer
  - Gradient clipping
  - Early stopping
- `predict(X)`: Forward pass through network
- `get_model_params()`: Returns `{'state_dict': state_dict, 'trained': bool}`
- `update_from_messages(messages)`: Aggregates state_dicts

#### Class: `GATAgent(Agent)`

**Model**: Graph Attention Network (PyTorch)

**Architecture**:
- Graph attention layers with multi-head attention
- Uses local graph structure (adjacency matrix)
- Aggregates neighbor features with attention weights

**Data Structures**:
- `local_graph`: Dict with `node_indices` and `adj_matrix`
- `full_graph`: Full adjacency matrix for reference

**Methods**: Similar to NeuralAgent but uses graph structure in forward pass.

#### Class: `RandomForestAgent(Agent)`

**Model**: scikit-learn `RandomForestClassifier`

**Parameters**:
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: 10)

**Data Structures**:
- Model parameters stored internally by sklearn
- Parameter extraction for aggregation is complex (not fully implemented)

**Methods**: Standard train/predict interface.

### Aggregation Strategies

### File: `src/agents/aggregation.py`

**Purpose**: Implements different strategies for combining parameters from neighbors.

#### Base Class: `AggregationStrategy(ABC)`

**Abstract Method**:
- `aggregate(messages, own_params)`: Combines parameters

#### Class: `AverageAggregation(AggregationStrategy)`

**Purpose**: Simple equal-weight averaging (FedAvg-style).

**Algorithm**:
1. Collect all valid parameters (from messages + own)
2. For linear models: Average `coef` and `intercept` arrays element-wise
3. For neural models: Average `state_dict` tensors element-wise
4. Return aggregated parameters

**Formula**: `θ_agg = (1/N) * Σ θ_i` where N = number of parameters

#### Class: `WeightedAverageAggregation(AggregationStrategy)`

**Purpose**: Weighted averaging based on edge weights.

**Algorithm**:
1. Extract weights from messages (edge weights)
2. Normalize weights to sum to 1.0
3. Add own parameters with weight = 1.0 / len(neighbors)
4. Re-normalize all weights
5. Compute weighted average: `θ_agg = Σ w_i * θ_i`

**Formula**: `θ_agg = (Σ w_i * θ_i) / (Σ w_i)`

**Note**: Own parameters get weight based on number of neighbors, not a fixed `w_self`.

#### Class: `AttentionAggregation(AggregationStrategy)`

**Purpose**: Attention-based aggregation using learned attention weights.

**Algorithm**:
1. Compute attention scores for each neighbor's parameters
2. Apply temperature scaling
3. Softmax to get attention weights
4. Weighted average using attention weights

**Implementation**: Uses neural network to compute attention scores based on parameter similarity.

#### Helper Function: `_normalize_coefficient_shapes(coef_list, intercept_list)`

**Purpose**: Handles shape mismatches when agents see different numbers of classes.

**Algorithm**:
1. Find maximum shape across all coefficients
2. Pad smaller arrays with zeros to match maximum shape
3. Handles 1D (binary) vs 2D (multiclass) cases

**Use Case**: When agents train on different class subsets, parameters have different shapes.

---

## Simulation Engine

### File: `src/simulation/engine.py`

**Purpose**: Core simulation engine that orchestrates multi-agent learning.

### Class: `SimulationEngine`

#### Initialization: `__init__(agents, network, test_data, test_labels, data_graph, test_indices)`

**Parameters**:
- `agents`: List[Agent], list of agent instances
- `network`: nx.DiGraph, communication network
- `test_data`: np.ndarray, test features
- `test_labels`: np.ndarray, test labels
- `data_graph`: Optional[nx.Graph], graph structure from dataset (for GAT agents)
- `test_indices`: Optional[np.ndarray], original indices of test samples

**Data Structures**:
```python
self.agents: List[Agent]
self.network: nx.DiGraph
self.test_data: np.ndarray          # Shape: (n_test, n_features)
self.test_labels: np.ndarray        # Shape: (n_test,)
self.data_graph: Optional[nx.Graph]
self.test_indices: np.ndarray       # Shape: (n_test,)
self.perturbations: List[Perturbation]
self.history: Dict[str, List] = {
    'accuracy': List[float],        # Accuracy per round
    'failed_nodes': List[int],       # Number of failed nodes per round
    'message_counts': List[int],     # Total messages per round
    'communication_costs': List[int] # Communication cost per round
}
self.current_time: int = 0
```

#### Method: `distribute_data(train_data, train_labels, distribution, samples_per_agent, train_indices)`

**Purpose**: Distributes training data across agents.

**Distribution Strategies**:

1. **'equal'** (default):
   - Divides data into N equal non-overlapping chunks
   - Agent i gets samples [i*samples_per_agent : (i+1)*samples_per_agent]
   - Last agent gets remaining samples

2. **'random'**:
   - Randomly permutes data indices
   - Divides into N equal chunks
   - Each agent gets random subset

3. **'biased'**:
   - Uses Dirichlet distribution to create non-uniform proportions
   - Some agents get more data than others

4. **'overlap'**:
   - Each agent gets overlapping data (30% overlap by default)
   - More data per agent, but with redundancy

**Algorithm**:
1. Calculate samples per agent based on strategy
2. For each agent:
   - Extract data subset
   - Call `agent.set_local_data(X_subset, y_subset)`
   - If graph-based agent, extract subgraph adjacency and call `agent.set_local_graph()`

**Graph Structure Handling**:
- For GAT/GNN agents, extracts subgraph adjacency matrix
- Maps original node indices to local indices
- Includes self-loops in adjacency

#### Method: `distribute_test_data(distribution='equal')`

**Purpose**: Distributes test data to agents (each agent only sees subset).

**Algorithm**: Similar to `distribute_data()` but for test set. Each agent predicts only on its assigned test subset.

**Use Case**: Enables distributed evaluation where agents make predictions on different test samples.

#### Method: `train_agents()`

**Purpose**: Performs initial local training on all agents.

**Algorithm**:
```python
for agent in self.agents:
    if agent.local_data is not None and not agent.failed:
        agent.train(agent.local_data, agent.local_labels)
```

**Result**: Each agent has trained model on its local data subset.

#### Method: `communicate() -> int`

**Purpose**: Executes one round of communication between agents.

**Algorithm**:
1. Iterate through all edges (u, v) in network
2. If agent u is not failed:
   - Get parameters from agent u: `params = agent[u].get_model_params()`
   - Get edge attributes: `weight`, `delay`
   - Create message: `{'params': params, 'weight': weight, 'sender': u, 'receiver': v}`
   - Calculate communication cost (parameter size)
   - Send message to agent v with delay: `agent[v].receive_message(message, delay)`
   - Set timestamp: `message_queue[-1]['timestamp'] = current_time`
3. Return total communication cost

**Message Structure**:
```python
message = {
    'params': {
        'coef': np.ndarray,        # For linear models
        'intercept': np.ndarray,   # For linear models
        'state_dict': Dict,        # For neural models
        'trained': bool
    },
    'weight': float,               # Edge weight (0.0-1.0)
    'sender': int,                 # Sender agent ID
    'receiver': int                # Receiver agent ID
}
```

**Communication Cost**: Counts parameter size (number of parameters sent). In practice, could be bytes or message count.

#### Method: `update_agents()`

**Purpose**: Processes messages and updates agent models.

**Algorithm**:
```python
for agent in self.agents:
    agent.process_messages(self.current_time)
```

**What happens**:
1. Agent checks message queue
2. Messages with `current_time >= timestamp + delay` are ready
3. Ready messages are passed to `agent.update_from_messages()`
4. Agent uses aggregation strategy to combine parameters
5. Agent updates its model with aggregated parameters

#### Method: `evaluate() -> float`

**Purpose**: Evaluates system-level accuracy using weighted majority voting.

**Algorithm**:
1. For each non-failed agent:
   - Agent predicts on its local test subset: `pred = agent.predict(agent.local_test_data)`
   - Calculate voting weight: `weight = degree + 1` (based on network degree)
   - For each test sample in agent's subset:
     - Store prediction with weight: `all_predictions[test_idx].append((pred, weight))`
     - Store true label: `all_labels[test_idx] = true_label`

2. For each test sample:
   - Collect all predictions from agents that saw this sample
   - Normalize weights: `weights = weights / weights.sum()`
   - Weighted voting: `votes = np.bincount(preds, weights=weights)`
   - Final prediction: `argmax(votes)`

3. Calculate accuracy: `mean(predictions == labels)`

**Weighted Voting Formula**:
```
For each class c:
    vote[c] = Σ(weight_i) for all agents i that predicted class c

Final prediction = argmax(vote)
```

**Why Weighted**: Agents with higher network degree (more connections) have more influence, reflecting their central role in information flow.

#### Method: `run(n_rounds: int, verbose: bool = True) -> Dict[str, Any]`

**Purpose**: Main simulation loop that runs for N communication rounds.

**Algorithm**:
```python
# Setup
set_full_graph_for_agents()        # For GAT agents
distribute_test_data()             # Distribute test data
train_agents()                     # Initial local training
evaluate()                         # Initial accuracy
record_history()                   # Record initial state

# Main loop
for round_num in range(n_rounds):
    current_time = round_num + 1
    
    # 1. Apply perturbations
    for perturbation in perturbations:
        perturbation.apply(agents, network, current_time)
    
    # 2. Communicate
    comm_cost = communicate()
    
    # 3. Update agents
    update_agents()
    
    # 4. Evaluate
    accuracy = evaluate()
    failed_count = count_failed_agents()
    message_count = count_messages()
    
    # 5. Record history
    history['accuracy'].append(accuracy)
    history['failed_nodes'].append(failed_count)
    history['message_counts'].append(message_count)
    history['communication_costs'].append(comm_cost)

# Return results
return get_results()
```

**Round Structure**:
1. **Perturbations**: Apply failures, delays, etc.
2. **Communication**: Agents send parameters to neighbors
3. **Aggregation**: Agents combine received parameters
4. **Evaluation**: System-level accuracy computed
5. **Recording**: Metrics stored in history

**Returns**: Dictionary with:
- `history`: All round-by-round metrics
- `final_accuracy`: Last round accuracy
- `average_accuracy`: Mean accuracy across rounds
- `accuracy_std`: Standard deviation
- `max_failed_nodes`: Maximum concurrent failures
- `total_messages`: Total messages exchanged
- `total_communication_cost`: Total communication cost
- `avg_communication_cost_per_round`: Average per round
- `network_info`: Network statistics

---

## Perturbations

### File: `src/simulation/perturbations.py`

**Purpose**: Implements various perturbations to test system robustness.

### Base Class: `Perturbation(ABC)`

**Abstract Methods**:
- `apply(agents, network, time_step)`: Apply perturbation
- `get_description() -> str`: Human-readable description

### Class: `NodeFailure(Perturbation)`

**Purpose**: Simulates random node failures with recovery.

**Parameters**:
- `failure_prob`: Probability of failure per round (default: 0.1)
- `failure_duration`: Rounds until recovery (default: 5)
- `start_time`: Round when failures begin (default: 5)
- `end_time`: Round when failures stop (optional)

**Data Structures**:
```python
self.failed_agents: Dict[int, int]  # agent_id -> recovery_time
```

**Algorithm**:
1. If `time_step < start_time` or `time_step > end_time`, return
2. Recover agents: For each failed agent, if `time_step >= recovery_time`, call `agent.recover()`
3. Fail agents: For each non-failed agent, with probability `failure_prob`, call `agent.fail()` and set `recovery_time = time_step + failure_duration`

**Failure Effects**:
- Failed agents cannot send messages
- Failed agents cannot receive messages
- Failed agents do not participate in evaluation
- Failed agents do not train

### Class: `DelayPerturbation(Perturbation)`

**Purpose**: Adds communication delays to edges.

**Parameters**:
- `delay_increase`: Additional delay to add (default: 2)
- `affected_prob`: Probability edge is affected (default: 0.3)
- `start_time`: When delays begin
- `end_time`: When delays end (optional, restores original)

**Data Structures**:
```python
self.original_delays: Dict[Tuple[int, int], int]  # (u, v) -> original_delay
```

**Algorithm**:
1. If `time_step < start_time`, return
2. If `time_step <= end_time`:
   - For each edge, with probability `affected_prob`, increase delay
   - Store original delay if not already stored
3. Else (after end_time):
   - Restore original delays

### Class: `WeightPerturbation(Perturbation)`

**Purpose**: Modifies edge weights (communication quality degradation).

**Parameters**:
- `weight_factor`: Multiplier for weights (default: 0.5)
- `affected_prob`: Probability edge is affected
- `start_time`, `end_time`: Time window

**Algorithm**: Similar to `DelayPerturbation` but modifies `weight` attribute instead of `delay`.

---

## Evaluation Metrics

### File: `src/evaluation/metrics.py`

**Purpose**: Computes various evaluation metrics for system analysis.

### Function: `calculate_accuracy(predictions, labels) -> float`

**Purpose**: Simple accuracy calculation.

**Algorithm**: `accuracy_score(predictions, labels)` from sklearn.

**Returns**: Accuracy as float (0.0-1.0).

### Function: `calculate_robustness(accuracy_history, failure_history, network) -> float`

**Purpose**: Computes comprehensive robustness score.

**Components**:

1. **Performance Resilience**:
   - Compare accuracy during failures vs. without failures
   - `performance_resilience = avg_accuracy_with_failures / avg_accuracy_without_failures`
   - Measures how well system maintains performance during failures

2. **Failure Impact**:
   - Average failure fraction: `avg_failure_fraction = mean(failures / n_agents)`
   - Failure frequency: `failure_frequency = count(failures > 0) / total_rounds`
   - Maximum failure fraction: `max_failure_fraction = max(failures) / n_agents`
   - Combined: `failure_impact = 0.4*avg + 0.3*freq + 0.3*max`

3. **Topology Resilience**:
   - Average shortest path length (if network connected)
   - Network density
   - `topology_resilience = f(path_length, density)`

4. **Stability**:
   - Coefficient of variation: `cv = std(accuracy) / mean(accuracy)`
   - `stability = 1.0 / (1.0 + 3.0 * cv)`

**Final Robustness**:
```python
if max(failures) == 0:
    robustness = stability * topology_resilience * 0.95
else:
    robustness = performance_resilience * (1.0 - 0.3*failure_impact) * topology_resilience * stability
```

**Returns**: Robustness score (0.0-1.0), higher is better.

### Function: `calculate_error_depth(agents, network, test_data, test_labels) -> Dict[str, float]`

**Purpose**: Analyzes error propagation through network.

**Algorithm**:
1. Identify incorrect predictions
2. For each error, trace back through network to find contributing agents
3. Calculate path length from error to source
4. Compute statistics: mean depth, max depth, error rate

**Returns**: Dictionary with `mean_depth`, `max_depth`, `error_rate`.

### Function: `calculate_failed_node_centrality(agents, network) -> Dict[str, float]`

**Purpose**: Analyzes centrality of failed nodes.

**Algorithm**:
1. Identify failed agents
2. Compute centrality metrics for failed nodes:
   - Degree centrality
   - Betweenness centrality
   - Closeness centrality
3. Compute averages

**Returns**: Dictionary with centrality metrics and failure rate.

### Function: `evaluate_system(agents, network, test_data, test_labels, accuracy_history, failure_history) -> Dict[str, Any]`

**Purpose**: Comprehensive system evaluation aggregator.

**Algorithm**:
1. Collect predictions from all agents (weighted majority voting)
2. Calculate accuracy
3. Calculate robustness using `calculate_robustness()`
4. Calculate error depth using `calculate_error_depth()`
5. Calculate failed node centrality using `calculate_failed_node_centrality()`

**Returns**: Dictionary with all metrics:
```python
{
    'accuracy': float,
    'robustness': float,
    'error_depth': Dict[str, float],
    'failed_node_centrality': Dict[str, float],
    'n_agents': int,
    'n_failed': int,
    'n_edges': int
}
```

---

## Visualization

### File: `src/utils/visualization.py`

**Purpose**: Generates visualizations for results analysis.

### Function: `plot_accuracy_history(history, save_path, title)`

**Purpose**: Plots accuracy and failure history over time.

**Output**: Two subplots:
- Top: Accuracy over rounds (line plot)
- Bottom: Failed nodes over rounds (line plot)

**Data**: Uses `history['accuracy']` and `history['failed_nodes']`.

### Function: `plot_network(network, agents, save_path, title)`

**Purpose**: Visualizes network topology.

**Algorithm**:
1. Use spring layout for node positions
2. Color nodes: red if failed, teal if active
3. Draw edges with width proportional to weight
4. Label nodes with agent IDs

**Output**: Network graph visualization.

### Function: `plot_comparison(results, save_path)`

**Purpose**: Compares multiple experiments.

**Output**: Bar chart comparing final accuracy across configurations.

### Function: `create_comprehensive_comparison(all_results, output_dir)`

**Purpose**: Creates comprehensive comparison plots for many experiments.

**Outputs**:
1. Accuracy heatmap (models × topologies)
2. Model performance bar chart (grouped bars, accuracy as %)
3. Robustness bar chart (grouped bars)
4. Average accuracy by model
5. Average accuracy by topology
6. Accuracy evolution by model
7. Accuracy evolution by topology
8. Best combinations (top 10 by accuracy and robustness)

**Data Processing**:
- Parses experiment names to extract model and topology
- Builds matrices: `accuracy_matrix[model_idx, topo_idx]`
- Aggregates across dimensions

### Function: `plot_robustness_and_communication_cost(all_results, output_dir)`

**Purpose**: Creates side-by-side bar charts for robustness and communication cost.

**Output**: Two bar charts:
- Left: Robustness by topology (averaged across models)
- Right: Communication cost by topology (averaged across models)

**Algorithm**:
1. Parse experiment names to extract topology and model
2. Group by topology
3. Average robustness and communication cost across models
4. Create bar charts

---

## Reporting

### File: `src/utils/reporting.py`

**Purpose**: Generates reports and serializes results.

### Function: `save_results_to_csv(results, output_path)`

**Purpose**: Flattens nested results dictionary and saves to CSV.

**Algorithm**:
1. Flatten nested dictionaries: `key_subkey_subsubkey = value`
2. Handle lists: compute mean, std, final value
3. Create pandas DataFrame
4. Save to CSV

**Data Structure**: Flattened dictionary with keys like:
- `final_accuracy`
- `history_accuracy_mean`
- `network_info_n_agents`
- `error_depth_mean_depth`

### Function: `generate_design_rules(all_results) -> List[str]`

**Purpose**: Automatically generates design recommendations from results.

**Algorithm**:
1. Extract metrics: accuracy, robustness, network size, density
2. Identify best configurations for different criteria
3. Compute correlations (e.g., size vs. accuracy)
4. Generate rule strings based on patterns

**Rules Generated**:
- Best for accuracy
- Best for robustness
- Scale impact (positive/negative/neutral)
- Failure resilience
- Communication efficiency
- General recommendations (star vs. cascade)

### Function: `generate_report(all_results, config, output_path)`

**Purpose**: Generates comprehensive markdown report.

**Sections**:
1. Executive Summary
2. Experimental Configuration
3. Results Summary (table)
4. Detailed Analysis (per configuration)
5. Design Rules & Recommendations
6. Conclusion

**Output**: Markdown file with tables, metrics, and recommendations.

---

## Detailed Data Flow

### Complete Execution Flow

#### Phase 1: Initialization

1. **Configuration Loading** (`cli.py::load_config()`)
   - Input: YAML file path
   - Process: Parse YAML → Python dict
   - Output: `config: dict`

2. **Dataset Loading** (`datasets/loaders.py::load_dataset()`)
   - Input: Dataset name, parameters
   - Process: 
     - Load graph structure (NetworkX)
     - Load/extract features (numpy array)
     - Load/extract labels (numpy array)
   - Output: `(graph: nx.Graph, features: np.ndarray, labels: np.ndarray)`
   - Data shapes:
     - `graph`: N nodes
     - `features`: (N, F) where F = feature count
     - `labels`: (N,)

3. **Data Splitting** (`cli.py::run_single_experiment()`)
   - Input: Features, labels
   - Process: `train_test_split()` with stratification
   - Output: `(train_idx, test_idx)`, `(X_train, y_train)`, `(X_test, y_test)`
   - Split: 70% train, 30% test (default)

4. **Topology Building** (`networks/topologies.py::build_topology()`)
   - Input: Topology name, n_agents, parameters
   - Process: Build NetworkX DiGraph with edges, weights, delays
   - Output: `network: nx.DiGraph`
   - Structure: N nodes, M edges (varies by topology)

5. **Agent Creation** (`cli.py::run_single_experiment()`)
   - Input: Agent type, n_features, n_classes, parameters
   - Process: Instantiate N agents of specified type
   - Output: `agents: List[Agent]`
   - Each agent: Has model, aggregation strategy, local data storage

#### Phase 2: Data Distribution

6. **Training Data Distribution** (`simulation/engine.py::distribute_data()`)
   - Input: X_train, y_train, distribution strategy
   - Process:
     - Divide data into N subsets (based on strategy)
     - For each agent i: `agent[i].set_local_data(X_subset, y_subset)`
     - For graph agents: Extract subgraph adjacency
   - Output: Each agent has local training data
   - Data transformation: `(N_train, F) → N subsets of size ~N_train/N`

7. **Test Data Distribution** (`simulation/engine.py::distribute_test_data()`)
   - Input: X_test, y_test
   - Process: Similar to training distribution
   - Output: Each agent has local test subset
   - Purpose: Distributed evaluation (each agent predicts on subset)

#### Phase 3: Initial Training

8. **Local Training** (`simulation/engine.py::train_agents()`)
   - Input: Agents with local data
   - Process: For each agent, `agent.train(agent.local_data, agent.local_labels)`
   - Output: Each agent has trained model
   - Model state: Parameters learned from local data only

9. **Initial Evaluation** (`simulation/engine.py::evaluate()`)
   - Input: Trained agents, test data
   - Process: Weighted majority voting across agents
   - Output: Initial accuracy (before communication)
   - Recorded in: `history['accuracy'][0]`

#### Phase 4: Communication Rounds

For each round `r` in `[1, n_rounds]`:

10. **Apply Perturbations** (`simulation/perturbations.py`)
    - Input: Agents, network, current_time
    - Process:
      - NodeFailure: Randomly fail agents with probability
      - DelayPerturbation: Add delays to edges
      - WeightPerturbation: Modify edge weights
    - Output: Modified agent states, network attributes

11. **Communication** (`simulation/engine.py::communicate()`)
    - Input: Agents, network
    - Process:
      - For each edge (u, v):
        - If agent u not failed:
          - Get parameters: `params = agent[u].get_model_params()`
          - Create message: `{'params': params, 'weight': weight, 'sender': u, 'receiver': v}`
          - Send to agent v: `agent[v].receive_message(message, delay)`
          - Set timestamp: `message_queue[-1]['timestamp'] = current_time`
    - Output: Messages queued in agents (with delays)
    - Communication cost: Counted and returned

12. **Message Processing** (`agents/base_agent.py::process_messages()`)
    - Input: Agent, current_time
    - Process:
      - Check message queue
      - Messages with `current_time >= timestamp + delay` are ready
      - Collect ready messages
      - If agent not failed: `agent.update_from_messages(ready_messages)`
    - Output: Ready messages processed

13. **Parameter Aggregation** (`agents/aggregation.py`)
    - Input: Messages (with parameters), own parameters
    - Process:
      - Extract parameters from messages
      - Apply aggregation strategy:
        - AverageAggregation: Equal-weight average
        - WeightedAverageAggregation: Weighted by edge weights
        - AttentionAggregation: Attention-weighted
      - Combine with own parameters
      - Normalize shapes if needed
    - Output: Aggregated parameters

14. **Model Update** (`agents/models.py::update_from_messages()`)
    - Input: Aggregated parameters
    - Process:
      - For linear models: Update `coef_` and `intercept_`
      - For neural models: Load `state_dict`
    - Output: Agent model updated with aggregated parameters

15. **Evaluation** (`simulation/engine.py::evaluate()`)
    - Input: Updated agents, test data
    - Process:
      - Each agent predicts on local test subset
      - Weighted majority voting (weights = degree + 1)
      - Calculate accuracy
    - Output: System accuracy for round r
    - Recorded in: `history['accuracy'][r]`

16. **History Recording** (`simulation/engine.py::run()`)
    - Record: accuracy, failed_nodes, message_counts, communication_costs
    - Stored in: `history` dictionary

#### Phase 5: Final Evaluation

17. **Comprehensive Evaluation** (`evaluation/metrics.py::evaluate_system()`)
    - Input: Agents, network, test data, accuracy_history, failure_history
    - Process:
      - Calculate final accuracy
      - Calculate robustness (multi-component)
      - Calculate error depth
      - Calculate failed node centrality
    - Output: Comprehensive metrics dictionary

#### Phase 6: Output Generation

18. **Results Serialization** (`utils/reporting.py::save_results_to_csv()`)
    - Input: Results dictionary
    - Process: Flatten nested structure → pandas DataFrame → CSV
    - Output: `{exp_name}_results.csv`

19. **Visualization** (`utils/visualization.py`)
    - Accuracy history: `plot_accuracy_history()` → `{exp_name}_accuracy.png`
    - Network topology: `plot_network()` → `{exp_name}_network.png`
    - Comparisons: `create_comprehensive_comparison()` → multiple plots
    - Robustness/communication: `plot_robustness_and_communication_cost()` → bar charts

20. **Report Generation** (`utils/reporting.py::generate_report()`)
    - Input: All results, configuration
    - Process: Generate markdown with tables, metrics, design rules
    - Output: `benchmark_report.md`

### Key Data Transformations

1. **YAML → Config Dict**: Nested structure with experiment parameters
2. **Dataset Files → Graph + Arrays**: Parsing and graph construction
3. **Full Dataset → Agent Subsets**: Data distribution (N → N subsets)
4. **Local Training → Model Parameters**: Training produces parameters
5. **Parameters → Messages**: Parameters packaged with metadata
6. **Messages → Aggregated Parameters**: Aggregation combines parameters
7. **Aggregated Parameters → Updated Models**: Models updated with aggregated values
8. **Agent Predictions → System Accuracy**: Weighted voting combines predictions
9. **Round History → Metrics**: Statistical analysis of history
10. **Results → Visualizations**: Plotting and visualization
11. **Results → Reports**: Markdown generation

### Key Algorithms

1. **FedAvg-style Aggregation**: Weighted averaging of model parameters
2. **Weighted Majority Voting**: Combining predictions with degree-based weights
3. **Message Queuing with Delays**: Time-based message delivery
4. **Graph Construction**: Various algorithms (Watts-Strogatz, Barabási-Albert, etc.)
5. **Robustness Calculation**: Multi-component metric combining performance, topology, stability
6. **Error Propagation Analysis**: Tracing errors through network structure

---

## Dependencies and External Libraries

### Core Dependencies

#### NetworkX (`networkx`)
- **Purpose**: Graph data structures and algorithms
- **Usage**:
  - Dataset graphs: `nx.Graph`, `nx.DiGraph`
  - Communication networks: `nx.DiGraph` with edge attributes
  - Graph algorithms: `nx.average_shortest_path_length()`, `nx.density()`, centrality measures
- **Key Classes**: `Graph`, `DiGraph`
- **Key Functions**: Graph construction, path analysis, centrality

#### NumPy (`numpy`)
- **Purpose**: Numerical operations and array handling
- **Usage**:
  - Feature arrays: `np.ndarray` for features and labels
  - Parameter arrays: Model coefficients, neural network weights
  - Statistical operations: `np.mean()`, `np.std()`, `np.bincount()`
- **Key Functions**: Array operations, statistical functions, random number generation

#### PyTorch (`torch`)
- **Purpose**: Deep learning models (Neural Networks, GAT)
- **Usage**:
  - Neural network layers: `nn.Linear`, `nn.BatchNorm1d`, `nn.Dropout`
  - Optimizers: `torch.optim.Adam`
  - Model state: `state_dict()` for parameter extraction
- **Key Classes**: `nn.Module`, `nn.Linear`, `torch.Tensor`
- **Key Functions**: Forward pass, backpropagation, parameter management

#### scikit-learn (`sklearn`)
- **Purpose**: Traditional machine learning models
- **Usage**:
  - Models: `LogisticRegression`, `RandomForestClassifier`, `Ridge`
  - Data generation: `make_classification()`, `make_moons()`
  - Preprocessing: `StandardScaler`, `train_test_split`
  - Metrics: `accuracy_score()`, `f1_score()`
- **Key Classes**: `LogisticRegression`, `RandomForestClassifier`
- **Key Functions**: `fit()`, `predict()`, `train_test_split()`

#### Matplotlib (`matplotlib`)
- **Purpose**: Plotting and visualization
- **Usage**:
  - Line plots: Accuracy history
  - Bar charts: Comparisons, robustness, communication cost
  - Heatmaps: Model × topology matrices
  - Network visualization: Node/edge drawing
- **Key Functions**: `plt.subplots()`, `plt.plot()`, `plt.bar()`, `plt.savefig()`

#### Seaborn (`seaborn`)
- **Purpose**: Statistical visualization
- **Usage**:
  - Heatmaps: `sns.heatmap()`
  - Styling: `sns.set_style()`, `sns.set_palette()`
- **Key Functions**: `heatmap()`, styling functions

#### YAML (`yaml`)
- **Purpose**: Configuration file parsing
- **Usage**: `yaml.safe_load()` to parse YAML config files
- **Key Functions**: `safe_load()`

#### Pandas (`pandas`)
- **Purpose**: Data manipulation and CSV export
- **Usage**:
  - DataFrame creation from results
  - CSV export: `df.to_csv()`
- **Key Classes**: `DataFrame`
- **Key Functions**: `to_csv()`

#### tqdm (`tqdm`)
- **Purpose**: Progress bars
- **Usage**: Progress indication during simulation rounds
- **Key Functions**: `tqdm()` for progress bars

### Dependency Relationships

```
Core Framework:
├── NetworkX (graph structures)
├── NumPy (arrays, numerical ops)
└── YAML (configuration)

ML Models:
├── scikit-learn (traditional ML)
└── PyTorch (neural networks)

Evaluation:
├── scikit-learn.metrics (accuracy, F1)
└── NetworkX (graph analysis)

Visualization:
├── Matplotlib (plotting)
└── Seaborn (statistical plots)

Utilities:
├── Pandas (data export)
└── tqdm (progress)
```

### Version Compatibility

- Python 3.7+
- NetworkX 2.5+
- NumPy 1.19+
- PyTorch 1.8+
- scikit-learn 0.24+
- Matplotlib 3.3+
- YAML: PyYAML 5.4+

---

## Design Decisions

### Why Message Queuing with Delays?

- **Realism**: Models network latency and asynchronous communication
- **Flexibility**: Allows modeling different delay scenarios
- **Controlled Testing**: Can inject delays to test robustness

### Why Weighted Majority Voting?

- **Reflects Importance**: More connected agents have more influence
- **Network-Aware**: Voting weights based on topology structure
- **Practical**: Simple but effective aggregation method

### Why Distributed Test Data?

- **Scalability**: Each agent only evaluates on subset
- **Realistic**: Mirrors distributed evaluation scenarios
- **Efficiency**: Reduces computation per agent

### Why Multiple Aggregation Strategies?

- **Flexibility**: Different strategies suit different scenarios
- **Research**: Enables comparison of aggregation methods
- **Extensibility**: Easy to add new strategies

### Why Separate Perturbation Classes?

- **Modularity**: Each perturbation type is independent
- **Composability**: Can combine multiple perturbations
- **Testability**: Easy to test individual perturbation effects

### Why Comprehensive Robustness Metric?

- **Multi-Dimensional**: Captures performance, topology, stability
- **Realistic**: Accounts for multiple failure scenarios
- **Comparable**: Provides single score for comparison

---

## Conclusion

This architecture documentation provides a comprehensive overview of the multi-agent network topology benchmarking framework. The system is designed with modularity, extensibility, and systematic evaluation in mind, enabling controlled experiments to study the effects of network topology on distributed learning performance.

For questions or contributions, refer to the source code in the `src/` directory and the configuration examples in `configs/`.

