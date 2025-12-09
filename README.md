# Multi-Agent AI Systems: Network Topology Benchmarking Framework

**Authors:** Kartikeya Gullapalli, Andres Wearden  
**Course:** ECE 381K - Multi-Agent Systems  
**University of Texas at Austin**  
**Date:** December 2024

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Usage](#3-usage)
4. [Configuration Parameters](#4-configuration-parameters)
5. [Source Code Documentation](#5-source-code-documentation)
6. [Output Files](#6-output-files)
7. [Reproducing Results](#7-reproducing-results)

---

## 1. Installation

### 1.1 Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows

### 1.2 Dependencies

Install all required packages:

```bash
pip install numpy scipy scikit-learn networkx matplotlib pandas pyyaml torch tqdm
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

#### Dependency List

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.21.0 | Numerical computations |
| `scipy` | ≥1.7.0 | Scientific computing |
| `scikit-learn` | ≥1.0.0 | Machine learning models (Logistic Regression, Random Forest) |
| `networkx` | ≥2.6.0 | Graph/network topology construction |
| `matplotlib` | ≥3.4.0 | Visualization and plotting |
| `pandas` | ≥1.3.0 | Data manipulation and CSV export |
| `pyyaml` | ≥5.4.0 | YAML configuration parsing |
| `torch` | ≥1.9.0 | Neural network and GAT implementations |
| `tqdm` | ≥4.62.0 | Progress bar display |

### 1.3 Verify Installation

```bash
python -c "from src.agents import LogisticAgent, GATAgent; print('Installation successful!')"
```

---

## 2. Quick Start

### Run a Quick Test

```bash
python cli.py --quick-test
```

This runs a small experiment with synthetic data to verify the system works.

### Run a Single Experiment

```bash
python cli.py --config configs/star_config.yaml --output outputs/my_experiment
```

### Run Full Benchmark (All Models × All Topologies)

```bash
python cli.py --config configs/all_models_all_topologies.yaml --output outputs/full_benchmark
```

---

## 3. Usage

### 3.1 Command Line Interface

```
python cli.py [OPTIONS]

OPTIONS:
  --config PATH     Path to YAML configuration file (required unless --quick-test)
  --output PATH     Output directory for results (default: outputs/)
  --quiet           Suppress verbose output
  --quick-test      Run quick test with synthetic data
  --help            Show help message
```

### 3.2 Examples

```bash
# Run logistic regression on 5 topologies
python cli.py --config configs/selected_topologies_logistic.yaml --output outputs/logistic_results

# Run GAT model on all topologies
python cli.py --config configs/all_topologies_gat.yaml --output outputs/gat_results

# Run neural network experiments
python cli.py --config configs/selected_topologies_neural.yaml --output outputs/neural_results

# Run random forest experiments
python cli.py --config configs/selected_topologies_rf.yaml --output outputs/rf_results

# Run comprehensive comparison (36 experiments)
python cli.py --config configs/all_models_all_topologies.yaml --output outputs/comprehensive
```

---

## 4. Configuration Parameters

Configuration files are in YAML format located in the `configs/` directory.

### 4.1 Top-Level Parameters

| Parameter | Type | Required | Description | Allowable Values |
|-----------|------|----------|-------------|------------------|
| `name` | string | Yes | Experiment identifier | Any string (no spaces recommended) |
| `dataset` | string | Yes | Dataset to use | `cora`, `synthetic_classification` |
| `topology` | string | Yes | Network topology type | See Section 4.2 |
| `n_agents` | integer | Yes | Number of agents | 2-100 (recommended: 5-20) |
| `agent_type` | string | Yes | Model type for agents | See Section 4.3 |
| `n_rounds` | integer | No | Number of training rounds | 1-100 (default: 20) |
| `test_size` | float | No | Fraction of data for testing | 0.1-0.5 (default: 0.3) |
| `data_distribution` | string | No | How to distribute data | `equal` (default) |

### 4.2 Topology Types and Parameters

#### Available Topologies

| Topology | Description |
|----------|-------------|
| `star` | Central hub connected to all agents |
| `cascade` | Sequential chain with optional skip connections |
| `feedback_rewired` | Ring with rewired feedback edges |
| `ring` | Circular connections |
| `mesh` | Fully connected network |
| `small_world` | Watts-Strogatz small-world network |
| `scale_free` | Barabási-Albert scale-free network |
| `tree` | Hierarchical tree structure |
| `grid` | 2D grid/lattice |

#### Topology-Specific Parameters (`topology_params`)

**Star Topology:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hub_id` | integer | 0 | ID of the central hub node (0 to n_agents-1) |
| `bidirectional` | boolean | true | Enable two-way communication |

**Cascade Topology:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_connections` | boolean | false | Add skip connections between non-adjacent nodes |
| `bidirectional` | boolean | false | Enable backward edges |
| `random_delays` | boolean | false | Add random communication delays |

**Feedback-Rewired Topology:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rewire_prob` | float | 0.3 | Probability of rewiring edges (0.0-1.0) |
| `feedback_prob` | float | 0.4 | Probability of adding feedback edges (0.0-1.0) |
| `base_topology` | string | "ring" | Base topology (`ring`, `mesh`, `random`) |

**Ring Topology:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bidirectional` | boolean | true | Enable two-way communication |

**Mesh Topology:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bidirectional` | boolean | true | Enable two-way communication |
| `edge_probability` | float | 1.0 | Probability of including each edge (0.0-1.0) |

**Small-World Topology:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | integer | 4 | Each node connects to k nearest neighbors (must be even, ≥2) |
| `rewire_prob` | float | 0.3 | Probability of rewiring each edge (0.0-1.0) |

**Scale-Free Topology:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `m` | integer | 2 | Number of edges to attach from new node (1 to n_agents-1) |

**Tree Topology:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `branching_factor` | integer | 2 | Number of children per node (≥1) |
| `bidirectional` | boolean | true | Enable two-way communication |

**Grid Topology:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rows` | integer | auto | Number of rows (rows × cols must equal n_agents) |
| `cols` | integer | auto | Number of columns |
| `bidirectional` | boolean | true | Enable two-way communication |

### 4.3 Agent Types and Parameters

#### Available Agent Types

| Agent Type | Description |
|------------|-------------|
| `logistic` | Logistic Regression classifier |
| `neural` | Multi-layer perceptron neural network |
| `gat` | Graph Attention Network |
| `rf` / `random_forest` | Random Forest ensemble |
| `gnn` | Basic Graph Neural Network |
| `graphsage` | GraphSAGE model |
| `gin` | Graph Isomorphism Network |

#### Agent-Specific Parameters (`agent_params`)

**Logistic Regression (`logistic`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `C` | float | 1.0 | Inverse regularization strength (>0) |
| `max_iter` | integer | 1000 | Maximum iterations for solver |

**Neural Network (`neural`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | integer | 128 | Hidden layer dimension |
| `n_layers` | integer | 3 | Number of hidden layers (1-10) |
| `lr` | float | 0.001 | Learning rate (0.0001-0.1) |
| `epochs` | integer | 100 | Training epochs per round |

**Graph Attention Network (`gat`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | integer | 64 | Hidden dimension |
| `num_heads` | integer | 2 | Number of attention heads (1-8) |
| `num_layers` | integer | 2 | Number of GAT layers (1-4) |
| `lr` | float | 0.01 | Learning rate |
| `epochs` | integer | 100 | Training epochs per round |
| `dropout` | float | 0.0 | Dropout probability (0.0-0.5) |

**Random Forest (`rf`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | integer | 100 | Number of trees (10-500) |
| `max_depth` | integer | 10 | Maximum tree depth (1-50, or null for unlimited) |
| `min_samples_split` | integer | 2 | Minimum samples to split node (2-20) |

### 4.4 Perturbation Parameters

Perturbations simulate failures during simulation.

**Node Failure (`node_failure`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `failure_prob` | float | 0.1 | Probability of node failure per round (0.0-1.0) |
| `failure_duration` | integer | 5 | Duration of failure in rounds |
| `start_time` | integer | 10 | Round when failures can start |

### 4.5 Example Configuration File

```yaml
experiments:
  - name: star_logistic_experiment
    dataset: cora
    topology: star
    n_agents: 10
    topology_params:
      hub_id: 0
      bidirectional: true
    agent_type: logistic
    agent_params:
      C: 1.0
      max_iter: 1000
    n_rounds: 30
    test_size: 0.3
    data_distribution: equal
    perturbations:
      - type: node_failure
        params:
          failure_prob: 0.1
          failure_duration: 5
          start_time: 10
```

---

## 5. Source Code Documentation

### 5.1 Project Structure

```
ece381k-project/
├── cli.py                 # Command-line interface (entry point)
├── configs/               # YAML configuration files
├── data/                  # Datasets (Cora)
├── outputs/               # Generated results and plots
├── src/                   # Source code modules
│   ├── agents/            # Agent implementations
│   │   ├── base_agent.py  # Abstract base class
│   │   ├── models.py      # Model implementations
│   │   └── aggregation.py # Aggregation strategies
│   ├── datasets/          # Data loading
│   │   └── loaders.py     # Dataset loaders
│   ├── networks/          # Network topologies
│   │   └── topologies.py  # Topology builders
│   ├── simulation/        # Simulation engine
│   │   ├── engine.py      # Main simulation loop
│   │   └── perturbations.py # Failure models
│   ├── evaluation/        # Metrics and evaluation
│   │   └── metrics.py     # Performance metrics
│   └── utils/             # Utilities
│       ├── visualization.py # Plotting functions
│       └── reporting.py   # Report generation
└── README.md              # This file
```

### 5.2 Core Algorithms

#### Distributed Learning Protocol

The simulation follows this algorithm each round:

```
Algorithm: Distributed Multi-Agent Learning
Input: agents[], network, train_data, test_data, n_rounds
Output: accuracy_history[]

1. Distribute train_data equally among agents
2. FOR round = 1 to n_rounds:
   a. Apply perturbations (node failures)
   b. FOR each agent in agents:
      - IF not failed: agent.train(local_data)
   c. FOR each edge (u, v) in network:
      - IF sender not failed: send(u.get_params(), v)
   d. FOR each agent in agents:
      - IF not failed: agent.aggregate(received_messages)
   e. predictions = ensemble_vote([agent.predict(test_data) for agent])
   f. accuracy_history.append(accuracy(predictions, test_labels))
3. RETURN accuracy_history
```

#### Model Aggregation

Agents aggregate received model parameters using averaging:

```
Algorithm: Parameter Averaging
Input: local_params, received_params[]
Output: updated_params

1. all_params = [local_params] + received_params
2. FOR each parameter p in param_names:
   updated_params[p] = mean([params[p] for params in all_params])
3. RETURN updated_params
```

### 5.3 Data Structures

#### Agent Class Hierarchy

```
Agent (base_agent.py)
├── LogisticAgent    - sklearn LogisticRegression wrapper
├── NeuralAgent      - PyTorch MLP implementation
├── GATAgent         - Graph Attention Network (PyTorch)
├── RandomForestAgent - sklearn RandomForest wrapper
├── GNNAgent         - Basic GNN
├── GraphSAGEAgent   - GraphSAGE model
└── GINAgent         - Graph Isomorphism Network
```

#### Network Representation

Networks are stored as `networkx.DiGraph` objects:
- **Nodes**: Integer IDs (0 to n_agents-1)
- **Edges**: Directed with attributes `weight` (float) and `delay` (int)

### 5.4 Key Classes and Functions

| Module | Class/Function | Description |
|--------|----------------|-------------|
| `cli.py` | `run_benchmark()` | Main entry point for experiments |
| `src/agents/models.py` | `LogisticAgent` | Logistic regression agent |
| `src/agents/models.py` | `GATAgent` | Graph Attention Network agent |
| `src/agents/models.py` | `NeuralAgent` | Neural network agent |
| `src/agents/models.py` | `RandomForestAgent` | Random forest agent |
| `src/networks/topologies.py` | `build_topology()` | Factory function for topologies |
| `src/simulation/engine.py` | `SimulationEngine` | Runs distributed learning simulation |
| `src/evaluation/metrics.py` | `evaluate_system()` | Computes performance metrics |

---

## 6. Output Files

After running an experiment, the following files are generated in the output directory:

| File Pattern | Description |
|--------------|-------------|
| `{experiment}_results.csv` | Raw results data |
| `{experiment}_accuracy.png` | Accuracy over rounds plot |
| `{experiment}_network.png` | Network topology visualization |
| `comparison.png` | Multi-experiment comparison plot |
| `benchmark_report.md` | Detailed markdown report |
| `accuracy_heatmap.png` | Model × Topology heatmap |
| `robustness_comparison.png` | Robustness metrics |

---

## 7. Reproducing Results

### 7.1 Main Results (4 Models × 5 Topologies)

To reproduce the main experimental results:

```bash
# Run Logistic Regression experiments
python cli.py --config configs/selected_topologies_logistic.yaml --output outputs/logistic

# Run GAT experiments
python cli.py --config configs/all_topologies_gat.yaml --output outputs/gat

# Run Neural Network experiments
python cli.py --config configs/selected_topologies_neural.yaml --output outputs/neural

# Run Random Forest experiments
python cli.py --config configs/selected_topologies_rf.yaml --output outputs/rf
```

### 7.2 Expected Results

| Model | Star | Cascade | Feedback | Mesh | Scale-Free |
|-------|------|---------|----------|------|------------|
| Logistic | ~63-70% | ~65-69% | ~65-69% | ~67-68% | ~65-68% |
| GAT | ~59-62% | ~63-66% | ~58-60% | ~64-67% | ~60-64% |
| RF | ~48-50% | ~48-50% | ~50-51% | ~48-49% | ~49-50% |
| Neural | ~30-40% | ~29-32% | ~39-48% | ~30-31% | ~31-32% |

*Note: Results may vary slightly due to random initialization and node failures.*

### 7.3 Full Benchmark

For comprehensive evaluation (36 experiments):

```bash
python cli.py --config configs/all_models_all_topologies.yaml --output outputs/full_benchmark
```

---

## Contact

- Kartikeya Gullapalli: kartikeya@utexas.edu
- Andres Wearden: andres.wearden@utexas.edu
