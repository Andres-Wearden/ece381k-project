# Multi-Agent AI Systems Benchmarking Framework

A comprehensive Python framework for simulating and benchmarking multi-agent AI systems as communication networks. This framework enables researchers and practitioners to evaluate how different network topologies, agent models, and perturbations affect collective intelligence and system robustness.

## Features

- **🤖 Multiple Agent Models**: Logistic regression, linear models, and small neural networks
- **🕸️ Nine Network Topologies**: Star, cascade, feedback-rewired, ring, mesh, small-world, scale-free, tree, and grid architectures
- **📊 Rich Datasets**: Karate Club, Cora, US Power Grid, and synthetic data generators
- **⚡ Perturbation System**: Node failures, communication delays, and weight perturbations
- **📈 Comprehensive Metrics**: Accuracy, robustness, error depth, and centrality analysis
- **📝 Auto-Generated Reports**: Markdown reports with design rules and insights
- **🎨 Visualization Tools**: Network graphs, accuracy plots, and comparison charts

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ECE381KProj

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run a Quick Test

```bash
python cli.py --quick-test
```

This will run a simple benchmark with default settings and save results to `outputs/`.

### Run with Configuration File

```bash
# Star topology
python cli.py --config configs/star_config.yaml

# Cascade topology
python cli.py --config configs/cascade_config.yaml

# Feedback-rewired topology
python cli.py --config configs/feedback_config.yaml

# Compare all topologies
python cli.py --config configs/all_topologies.yaml --output results/comparison
```

### Programmatic Usage

```python
from src.datasets import load_dataset
from src.agents import LogisticAgent
from src.networks import build_topology
from src.simulation import SimulationEngine, NodeFailure

# Load dataset
graph, features, labels = load_dataset('karate')

# Build network
network = build_topology('star', n_agents=10)

# Create agents
agents = [LogisticAgent(i, features.shape[1], len(np.unique(labels))) 
          for i in range(10)]

# Run simulation
sim = SimulationEngine(agents, network, X_test, y_test)
sim.distribute_data(X_train, y_train)
sim.add_perturbation(NodeFailure(failure_prob=0.1))
results = sim.run(n_rounds=20)

print(f"Final Accuracy: {results['final_accuracy']:.4f}")
```

See `examples/run_benchmark.py` for complete examples.

## Architecture

```
ECE381KProj/
├── src/
│   ├── agents/          # Agent models (logistic, linear, neural)
│   ├── networks/        # Topology builders (9 topologies: star, cascade, feedback, ring, mesh, small_world, scale_free, tree, grid)
│   ├── datasets/        # Dataset loaders
│   ├── simulation/      # Simulation engine and perturbations
│   ├── evaluation/      # Metrics (accuracy, robustness, etc.)
│   └── utils/           # Reporting and visualization
├── configs/             # YAML configuration files
├── examples/            # Example scripts
├── outputs/             # Generated results (created automatically)
├── cli.py               # Command-line interface
└── requirements.txt     # Python dependencies
```

## Network Topologies

### 1. Star Topology
- Central hub connected to all agents
- Efficient information aggregation
- Single point of failure (hub)
- Best for: Centralized coordination, fast consensus

```python
network = build_topology('star', n_agents=10, hub_id=0, bidirectional=True)
```

### 2. Cascade Topology
- Sequential chain of agents
- Information flows in stages
- Supports skip connections
- Best for: Sequential processing, staged refinement

```python
network = build_topology('cascade', n_agents=10, 
                        skip_connections=True, bidirectional=True)
```

### 3. Feedback-Rewired Topology
- Dynamic network with feedback loops
- Rewired connections for exploration
- Rich information flow patterns
- Best for: Robustness, complex tasks

```python
network = build_topology('feedback_rewired', n_agents=10,
                        rewire_prob=0.3, feedback_prob=0.4,
                        base_topology='ring')
```

### 4. Ring Topology
- Circular connections, each node connected to neighbors
- Decentralized structure
- Low connectivity, high path length
- Best for: Decentralized systems, equal participation

```python
network = build_topology('ring', n_agents=10, bidirectional=True)
```

### 5. Mesh/Fully Connected Topology
- All nodes connected to all others
- Maximum connectivity
- Fastest information propagation
- Best for: Small networks, maximum consensus speed

```python
network = build_topology('mesh', n_agents=10, bidirectional=True, edge_probability=1.0)
```

### 6. Small-World Topology
- Watts-Strogatz model
- High clustering with short path lengths
- Realistic network properties
- Best for: Real-world network simulation, balanced connectivity

```python
network = build_topology('small_world', n_agents=10, k=4, rewire_prob=0.3)
```

### 7. Scale-Free Topology
- Barabási–Albert model
- Power-law degree distribution
- Hub nodes with many connections
- Best for: Modeling real networks (social, internet), robustness analysis

```python
network = build_topology('scale_free', n_agents=10, m=2)
```

### 8. Tree Topology
- Hierarchical structure
- Root-to-leaf information flow
- Balanced branching
- Best for: Hierarchical organizations, structured data flow

```python
network = build_topology('tree', n_agents=10, branching_factor=2, bidirectional=True)
```

### 9. Grid/Lattice Topology
- 2D regular grid structure
- Regular connectivity pattern
- Spatial organization
- Best for: Spatial tasks, regular communication patterns

```python
network = build_topology('grid', n_agents=10, bidirectional=True)
```

## Agent Models

### Logistic Regression Agent
```python
agent = LogisticAgent(agent_id=0, n_features=10, n_classes=2, C=1.0)
```

### Linear Regression Agent
```python
agent = LinearAgent(agent_id=0, n_features=10, n_classes=2, alpha=1.0)
```

### Neural Network Agent
```python
agent = NeuralAgent(agent_id=0, n_features=10, n_classes=2, 
                   hidden_dim=32, lr=0.01, epochs=50)
```

## Datasets

### Built-in Datasets
- **Karate Club**: Social network (34 nodes)
- **Cora**: Citation network (2,708 papers, 1,433 features, 7 classes) - Full real dataset
- **Power Grid**: Infrastructure network (150 nodes)

### Synthetic Datasets
- **Classification**: Configurable features and classes
- **Moons**: Non-linear decision boundary

```python
# Load built-in dataset
graph, features, labels = load_dataset('karate')

# Generate synthetic dataset
graph, features, labels = load_dataset('synthetic_classification',
                                      n_nodes=100, n_features=15, n_classes=3)
```

## Perturbations

### Node Failures
```python
perturbation = NodeFailure(
    failure_prob=0.1,       # 10% probability per round
    failure_duration=5,     # Recover after 5 rounds
    start_time=10,          # Start at round 10
    end_time=20            # Stop at round 20
)
sim.add_perturbation(perturbation)
```

### Communication Delays
```python
perturbation = DelayPerturbation(
    delay_increase=2,       # Add 2 round delay
    affected_prob=0.3,      # Affect 30% of edges
    start_time=5
)
sim.add_perturbation(perturbation)
```

### Weight Perturbations
```python
perturbation = WeightPerturbation(
    weight_factor=0.5,      # Reduce weights to 50%
    affected_prob=0.3,
    start_time=5
)
sim.add_perturbation(perturbation)
```

## Evaluation Metrics

The framework computes comprehensive metrics:

1. **Accuracy**: Classification accuracy on test data
2. **Robustness**: Stability under perturbations
3. **Error Depth**: Average propagation distance of errors
4. **Failed Node Centrality**: Impact of failed nodes based on centrality
5. **Network Metrics**: Density, connectivity, message counts

## Configuration Files

Configuration files use YAML format:

```yaml
name: my_experiment
dataset: synthetic_classification
dataset_params:
  n_nodes: 100
  n_features: 15
  n_classes: 3

topology: star
n_agents: 15
topology_params:
  hub_id: 0
  bidirectional: true

agent_type: logistic
agent_params:
  C: 1.0
  max_iter: 1000

n_rounds: 25
test_size: 0.3
data_distribution: equal

perturbations:
  - type: node_failure
    params:
      failure_prob: 0.1
      failure_duration: 5
      start_time: 10
```

## Output

The framework generates:

1. **CSV Files**: Detailed metrics for each experiment
2. **Plots**:
   - Accuracy history over time
   - Network topology visualization
   - Comparison charts across topologies
   - Metrics heatmaps
3. **Markdown Report**: Comprehensive analysis with:
   - Results summary table
   - Detailed metrics per topology
   - Design rules and recommendations
   - Performance insights

Example output structure:
```
outputs/
├── star_topology_results.csv
├── star_topology_accuracy.png
├── star_topology_network.png
├── cascade_topology_results.csv
├── cascade_topology_accuracy.png
├── comparison.png
├── accuracy_comparison.png
├── metrics_heatmap.png
└── benchmark_report.md
```

## Design Rules

The framework automatically generates design rules based on experimental results:

- **When to Use Star**: Best accuracy for centralized coordination
- **When to Use Cascade**: Sequential processing benefits
- **When to Use Feedback-Rewired**: Maximum robustness under failures
- **Scaling Guidelines**: How network size affects performance
- **Communication Efficiency**: Balance between connectivity and overhead

## Examples

### Example 1: Basic Star Topology

```python
from src.datasets import load_dataset
from src.agents import LogisticAgent
from src.networks import build_topology
from src.simulation import SimulationEngine

# Setup
graph, X, y = load_dataset('karate')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create network and agents
network = build_topology('star', n_agents=10)
agents = [LogisticAgent(i, X.shape[1], len(np.unique(y))) for i in range(10)]

# Simulate
sim = SimulationEngine(agents, network, X_test, y_test)
sim.distribute_data(X_train, y_train)
results = sim.run(n_rounds=20)
```

### Example 2: Compare Agent Types

```python
for agent_type in ['logistic', 'linear', 'neural']:
    agents = create_agents(agent_type, n_agents=10)
    sim = SimulationEngine(agents, network, X_test, y_test)
    sim.distribute_data(X_train, y_train)
    results = sim.run(n_rounds=20)
    print(f"{agent_type}: {results['final_accuracy']:.4f}")
```

### Example 3: Custom Topology

```python
import networkx as nx

# Create custom directed graph
G = nx.DiGraph()
G.add_nodes_from(range(10))
# Add custom edges with weights and delays
G.add_edge(0, 1, weight=0.8, delay=0)
G.add_edge(1, 2, weight=0.9, delay=1)
# ... add more edges

# Use in simulation
sim = SimulationEngine(agents, G, X_test, y_test)
```

## CLI Reference

```bash
# Basic usage
python cli.py --config <config_file> [options]

# Options:
--config PATH       Path to YAML configuration file
--output DIR        Output directory (default: outputs)
--quiet            Suppress verbose output
--quick-test       Run quick test with defaults

# Examples:
python cli.py --quick-test
python cli.py --config configs/star_config.yaml
python cli.py --config configs/all_topologies.yaml --output results/exp1
python cli.py --config configs/feedback_config.yaml --quiet
```

## Advanced Usage

### Custom Agent Implementation

```python
from src.agents.base_agent import Agent

class MyCustomAgent(Agent):
    def train(self, X, y):
        # Implement training logic
        pass
    
    def predict(self, X):
        # Implement prediction logic
        pass
    
    def get_model_params(self):
        # Return parameters to share
        pass
    
    def update_from_messages(self, messages):
        # Update based on received messages
        pass
```

### Custom Perturbation

```python
from src.simulation.perturbations import Perturbation

class MyPerturbation(Perturbation):
    def apply(self, agents, network, time_step):
        # Implement perturbation logic
        pass
    
    def get_description(self):
        return "My custom perturbation"
```

## Dependencies

- Python 3.8+
- NetworkX >= 3.1
- NumPy >= 1.24
- Pandas >= 2.0
- Matplotlib >= 3.7
- Seaborn >= 0.12
- Scikit-learn >= 1.3
- PyTorch >= 2.0 (for neural agents)
- PyYAML >= 6.0

## License

MIT License

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{multiagent_benchmark_2024,
  title={Multi-Agent AI Systems Benchmarking Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ECE381KProj}
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Contact

For questions or support, please open an issue on GitHub.

---

**Happy Benchmarking! 🚀**

