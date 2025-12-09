# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-12-08 17:27:01

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
- **Topologies Tested**: star_topology_rf, cascade_topology_rf, feedback_rewired_topology_rf, mesh_topology_rf, scale_free_topology_rf

## Results Summary

| Topology | Final Accuracy | Avg Accuracy | Robustness | Error Depth | Network Density |
|----------|----------------|--------------|------------|-------------|-----------------|
| star_topology_rf | 0.4815 | 0.4911 | 0.9983 | 1.7000 | 0.2000 |
| cascade_topology_rf | 0.4826 | 0.4990 | 0.9999 | 2.9500 | 0.2778 |
| feedback_rewired_topology_rf | 0.5088 | 0.4932 | 0.9999 | 2.3929 | 0.1222 |
| mesh_topology_rf | 0.4853 | 0.4882 | 0.9941 | 0.9000 | 1.0000 |
| scale_free_topology_rf | 0.4947 | 0.4968 | 0.9997 | 1.4714 | 0.3556 |

## Detailed Analysis

### star_topology_rf

**Performance Metrics:**
- Final Accuracy: 0.4815
- Average Accuracy: 0.4911
- Accuracy Std Dev: 0.0163
- Robustness Score: 0.9983

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 8
- Total Messages Exchanged: 316

**Error Propagation:**
- Mean Error Depth: 1.7000
- Max Error Depth: 1.7000
- Error Rate: 0.5264

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.1250
- Failure Rate: 0.8000

### cascade_topology_rf

**Performance Metrics:**
- Final Accuracy: 0.4826
- Average Accuracy: 0.4990
- Accuracy Std Dev: 0.0116
- Robustness Score: 0.9999

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 25
- Network Density: 0.2778

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 457

**Error Propagation:**
- Mean Error Depth: 2.9500
- Max Error Depth: 4.5000
- Error Rate: 0.5375

**Failed Node Impact:**
- Avg Degree Centrality: 0.3333
- Avg Betweenness Centrality: 0.2523
- Failure Rate: 0.4000

### feedback_rewired_topology_rf

**Performance Metrics:**
- Final Accuracy: 0.5088
- Average Accuracy: 0.4932
- Accuracy Std Dev: 0.0114
- Robustness Score: 0.9999

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 11
- Network Density: 0.1222

**Reliability:**
- Max Failed Nodes: 3
- Total Messages Exchanged: 244

**Error Propagation:**
- Mean Error Depth: 2.3929
- Max Error Depth: 4.5000
- Error Rate: 0.5092

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.1667
- Failure Rate: 0.3000

### mesh_topology_rf

**Performance Metrics:**
- Final Accuracy: 0.4853
- Average Accuracy: 0.4882
- Accuracy Std Dev: 0.0155
- Robustness Score: 0.9941

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 1548

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.5474

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.5000

### scale_free_topology_rf

**Performance Metrics:**
- Final Accuracy: 0.4947
- Average Accuracy: 0.4968
- Accuracy Std Dev: 0.0172
- Robustness Score: 0.9997

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 32
- Network Density: 0.3556

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 712

**Error Propagation:**
- Mean Error Depth: 1.4714
- Max Error Depth: 1.7000
- Error Rate: 0.5351

**Failed Node Impact:**
- Avg Degree Centrality: 0.2593
- Avg Betweenness Centrality: 0.0201
- Failure Rate: 0.3000

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: feedback_rewired_topology_rf achieved 0.509 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: feedback_rewired_topology_rf achieved 1.000 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. 💪 **Failure Resilience**: star_topology_rf maintains performance best under node failures. This topology has effective redundancy and graceful degradation.

5. ⚡ **Communication Efficient**: feedback_rewired_topology_rf, scale_free_topology_rf achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **feedback_rewired_topology_rf** topology achieved the highest final accuracy of 0.5088. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*