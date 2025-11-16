# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-11-16 17:18:05

---

## Executive Summary

This report presents a comprehensive evaluation of multi-agent AI systems 
across different network topologies. The benchmark assesses accuracy, robustness, 
error propagation, and failure resilience.

## Experimental Configuration

- **Dataset**: N/A
- **Agent Model**: N/A
- **Number of Agents**: N/A
- **Communication Rounds**: N/A
- **Topologies Tested**: star_topology_neural, cascade_topology_neural, feedback_rewired_topology_neural, mesh_topology_neural, scale_free_topology_neural

## Results Summary

| Topology | Final Accuracy | Avg Accuracy | Robustness | Error Depth | Network Density |
|----------|----------------|--------------|------------|-------------|-----------------|
| star_topology_neural | 0.3149 | 0.3217 | 0.9561 | 1.5667 | 0.2000 |
| cascade_topology_neural | 0.3056 | 0.3077 | 0.9618 | 2.5750 | 0.2333 |
| feedback_rewired_topology_neural | 0.3917 | 0.3622 | 0.9794 | 2.7806 | 0.1667 |
| mesh_topology_neural | 0.3026 | 0.3146 | 0.9831 | 0.9000 | 1.0000 |
| scale_free_topology_neural | 0.3104 | 0.3141 | 0.9787 | 1.5571 | 0.3556 |

## Detailed Analysis

### star_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.3149
- Average Accuracy: 0.3217
- Accuracy Std Dev: 0.0521
- Robustness Score: 0.9561

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 316

**Error Propagation:**
- Mean Error Depth: 1.5667
- Max Error Depth: 1.7000
- Error Rate: 0.6937

**Failed Node Impact:**
- Avg Degree Centrality: 0.1111
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.4000

### cascade_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.3056
- Average Accuracy: 0.3077
- Accuracy Std Dev: 0.0452
- Robustness Score: 0.9618

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 21
- Network Density: 0.2333

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 393

**Error Propagation:**
- Mean Error Depth: 2.5750
- Max Error Depth: 3.7000
- Error Rate: 0.6974

**Failed Node Impact:**
- Avg Degree Centrality: 0.2593
- Avg Betweenness Centrality: 0.2824
- Failure Rate: 0.6000

### feedback_rewired_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.3917
- Average Accuracy: 0.3622
- Accuracy Std Dev: 0.0529
- Robustness Score: 0.9794

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 15
- Network Density: 0.1667

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 290

**Error Propagation:**
- Mean Error Depth: 2.7806
- Max Error Depth: 3.5556
- Error Rate: 0.6765

**Failed Node Impact:**
- Avg Degree Centrality: 0.1667
- Avg Betweenness Centrality: 0.1076
- Failure Rate: 0.2000

### mesh_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.3026
- Average Accuracy: 0.3146
- Accuracy Std Dev: 0.0424
- Robustness Score: 0.9831

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 1670

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.6974

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### scale_free_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.3104
- Average Accuracy: 0.3141
- Accuracy Std Dev: 0.0415
- Robustness Score: 0.9787

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 32
- Network Density: 0.3556

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 568

**Error Propagation:**
- Mean Error Depth: 1.5571
- Max Error Depth: 1.9000
- Error Rate: 0.6974

**Failed Node Impact:**
- Avg Degree Centrality: 0.3704
- Avg Betweenness Centrality: 0.1049
- Failure Rate: 0.3000

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: feedback_rewired_topology_neural achieved 0.392 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: mesh_topology_neural achieved 0.983 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. 💪 **Failure Resilience**: mesh_topology_neural maintains performance best under node failures. This topology has effective redundancy and graceful degradation.

5. ⚡ **Communication Efficient**: feedback_rewired_topology_neural achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **feedback_rewired_topology_neural** topology achieved the highest final accuracy of 0.3917. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*