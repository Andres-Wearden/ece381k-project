# Multi-Agent Network Topology Benchmarking

**ECE 381K - Multi-Agent Systems**  
**Authors:** Kartikeya Gullapalli, Andres Wearden

---

## Installation

```bash
pip install numpy scipy scikit-learn networkx matplotlib pandas pyyaml torch tqdm
```

---

## Running Experiments

### Quick Test
```bash
python cli.py --quick-test
```

### Run All Models on All Topologies (Full Benchmark)
```bash
python cli.py --config configs/all_models_all_topologies.yaml --output outputs/results
```

### Run Individual Model Types
```bash
# Logistic Regression
python cli.py --config configs/selected_topologies_logistic.yaml --output outputs/logistic

# Graph Attention Network
python cli.py --config configs/all_topologies_gat.yaml --output outputs/gat

# Neural Network
python cli.py --config configs/selected_topologies_neural.yaml --output outputs/neural

# Random Forest
python cli.py --config configs/selected_topologies_rf.yaml --output outputs/rf
```

---

## Configuration Parameters

Configuration files are YAML files in `configs/`. Example:

```yaml
experiments:
  - name: my_experiment
    dataset: cora                    # Dataset: "cora" or "synthetic_classification"
    topology: star                   # See topology options below
    n_agents: 10                     # Number of agents: 2-100
    agent_type: logistic             # See agent options below
    n_rounds: 30                     # Training rounds: 1-100
    test_size: 0.3                   # Test split: 0.1-0.5
```

### Topology Options

| Topology | Parameters |
|----------|------------|
| `star` | `hub_id` (0 to n-1), `bidirectional` (true/false) |
| `cascade` | `skip_connections` (true/false), `bidirectional` (true/false) |
| `feedback_rewired` | `rewire_prob` (0.0-1.0), `feedback_prob` (0.0-1.0) |
| `ring` | `bidirectional` (true/false) |
| `mesh` | `edge_probability` (0.0-1.0) |
| `small_world` | `k` (neighbors, 2-10), `rewire_prob` (0.0-1.0) |
| `scale_free` | `m` (edges per new node, 1-5) |
| `tree` | `branching_factor` (children per node, 1-5) |
| `grid` | `rows`, `cols` (must multiply to n_agents) |

### Agent Options

| Agent | Parameters |
|-------|------------|
| `logistic` | `C` (regularization, 0.01-10.0), `max_iter` (100-5000) |
| `neural` | `hidden_dim` (32-256), `n_layers` (1-10), `lr` (0.0001-0.1), `epochs` (10-200) |
| `gat` | `hidden_dim` (32-128), `num_heads` (1-8), `num_layers` (1-4), `lr` (0.001-0.1) |
| `rf` | `n_estimators` (10-500), `max_depth` (1-50) |

### Node Failure Perturbations

```yaml
perturbations:
  - type: node_failure
    params:
      failure_prob: 0.1      # Probability per round: 0.0-1.0
      failure_duration: 5    # Rounds failed: 1-20
      start_time: 10         # Round to start: 0-n_rounds
```

---

## Output

Results are saved to the output directory:
- `*_results.csv` - Raw data
- `*_accuracy.png` - Accuracy plots
- `*_network.png` - Network visualizations
- `benchmark_report.md` - Summary report

---

## Project Structure

```
src/
├── agents/       # Agent models (LogisticAgent, GATAgent, NeuralAgent, etc.)
├── networks/     # Topology builders (star, cascade, mesh, etc.)
├── simulation/   # Simulation engine and perturbations
├── evaluation/   # Metrics computation
└── utils/        # Visualization and reporting
```

---

## Source Code Documentation

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and numerical computations |
| `torch` | Neural network and GAT model implementations |
| `scikit-learn` | Logistic Regression and Random Forest models |
| `networkx` | Graph data structure for network topologies |
| `matplotlib` | Plotting accuracy curves and network visualizations |
| `pandas` | CSV export and data manipulation |
| `pyyaml` | Parsing YAML configuration files |
| `tqdm` | Progress bar for simulation |

### Algorithms

**Distributed Learning (src/simulation/engine.py):**
```
For each round:
  1. Each agent trains on its local data subset
  2. Agents send model parameters to neighbors (based on topology edges)
  3. Each agent averages received parameters with its own (FedAvg-style)
  4. Evaluate ensemble prediction on test set
```

**Parameter Aggregation (src/agents/base_agent.py):**
- Receives model parameters from neighboring agents
- Computes element-wise average of all parameters (local + received)
- Updates local model with averaged parameters

**Topology Construction (src/networks/topologies.py):**
- Star: Hub-and-spoke pattern using central node
- Cascade: Chain with optional skip connections
- Small-world: Watts-Strogatz algorithm (ring + random rewiring)
- Scale-free: Barabási-Albert preferential attachment
- Mesh: Complete graph (all-to-all connections)

### Data Structures

| Structure | Location | Description |
|-----------|----------|-------------|
| `Agent` | `src/agents/base_agent.py` | Base class with `train()`, `predict()`, `get_params()`, `aggregate()` methods |
| `nx.DiGraph` | `src/networks/topologies.py` | Directed graph storing agent connections with edge weights |
| `SimulationEngine` | `src/simulation/engine.py` | Manages agents, network, data distribution, and training loop |
| `NodeFailure` | `src/simulation/perturbations.py` | Simulates random agent failures during training |

### Key Files

| File | Description |
|------|-------------|
| `cli.py` | Entry point - parses args, loads config, runs experiments |
| `src/agents/models.py` | All agent implementations (Logistic, Neural, GAT, RF) |
| `src/networks/topologies.py` | All topology builders (star, cascade, mesh, etc.) |
| `src/simulation/engine.py` | Main simulation loop and data distribution |
| `src/evaluation/metrics.py` | Accuracy, robustness, and error propagation metrics |
