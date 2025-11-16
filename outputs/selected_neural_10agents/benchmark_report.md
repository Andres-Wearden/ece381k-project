# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-11-16 17:14:12

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
| star_topology_neural | 0.1474 | 0.1607 | 0.9708 | 1.5857 | 0.2000 |
| cascade_topology_neural | 0.1506 | 0.1600 | 0.9814 | 2.6200 | 0.2444 |
| feedback_rewired_topology_neural | 0.2646 | 0.2316 | 0.9992 | 2.0833 | 0.1333 |
| mesh_topology_neural | 0.1564 | 0.1597 | 0.9862 | 0.9000 | 1.0000 |
| scale_free_topology_neural | 0.1570 | 0.1651 | 0.9838 | 1.4857 | 0.3556 |

## Detailed Analysis

### star_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.1474
- Average Accuracy: 0.1607
- Accuracy Std Dev: 0.0353
- Robustness Score: 0.9708

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 8
- Total Messages Exchanged: 342

**Error Propagation:**
- Mean Error Depth: 1.5857
- Max Error Depth: 1.7000
- Error Rate: 0.8450

**Failed Node Impact:**
- Avg Degree Centrality: 0.1111
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.3000

### cascade_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.1506
- Average Accuracy: 0.1600
- Accuracy Std Dev: 0.0271
- Robustness Score: 0.9814

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 22
- Network Density: 0.2444

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 464

**Error Propagation:**
- Mean Error Depth: 2.6200
- Max Error Depth: 3.7000
- Error Rate: 0.8426

**Failed Node Impact:**
- Avg Degree Centrality: 0.2667
- Avg Betweenness Centrality: 0.2389
- Failure Rate: 0.5000

### feedback_rewired_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.2646
- Average Accuracy: 0.2316
- Accuracy Std Dev: 0.0202
- Robustness Score: 0.9992

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 12
- Network Density: 0.1333

**Reliability:**
- Max Failed Nodes: 4
- Total Messages Exchanged: 262

**Error Propagation:**
- Mean Error Depth: 2.0833
- Max Error Depth: 3.0000
- Error Rate: 0.7122

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.1019
- Failure Rate: 0.3000

### mesh_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.1564
- Average Accuracy: 0.1597
- Accuracy Std Dev: 0.0249
- Robustness Score: 0.9862

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 1842

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.8450

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.4000

### scale_free_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.1570
- Average Accuracy: 0.1651
- Accuracy Std Dev: 0.0288
- Robustness Score: 0.9838

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 32
- Network Density: 0.3556

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 644

**Error Propagation:**
- Mean Error Depth: 1.4857
- Max Error Depth: 1.9000
- Error Rate: 0.8426

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.0077
- Failure Rate: 0.3000

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: feedback_rewired_topology_neural achieved 0.265 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: feedback_rewired_topology_neural achieved 0.999 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. 💪 **Failure Resilience**: star_topology_neural maintains performance best under node failures. This topology has effective redundancy and graceful degradation.

5. ⚡ **Communication Efficient**: feedback_rewired_topology_neural achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **feedback_rewired_topology_neural** topology achieved the highest final accuracy of 0.2646. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*