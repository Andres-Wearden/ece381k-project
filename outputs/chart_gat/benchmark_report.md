# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-12-08 17:28:33

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
- **Topologies Tested**: star_topology_gat, cascade_topology_gat, feedback_rewired_topology_gat, mesh_topology_gat, scale_free_topology_gat

## Results Summary

| Topology | Final Accuracy | Avg Accuracy | Robustness | Error Depth | Network Density |
|----------|----------------|--------------|------------|-------------|-----------------|
| star_topology_gat | 0.5965 | 0.5882 | 0.9997 | 1.7000 | 0.2000 |
| cascade_topology_gat | 0.6368 | 0.6240 | 0.9996 | 2.7000 | 0.2556 |
| feedback_rewired_topology_gat | 0.5808 | 0.5841 | 0.9886 | 1.9802 | 0.1667 |
| mesh_topology_gat | 0.6420 | 0.6315 | 0.9993 | 0.9000 | 1.0000 |
| scale_free_topology_gat | 0.5975 | 0.6017 | 0.9998 | 1.5500 | 0.3556 |

## Detailed Analysis

### star_topology_gat

**Performance Metrics:**
- Final Accuracy: 0.5965
- Average Accuracy: 0.5882
- Accuracy Std Dev: 0.0184
- Robustness Score: 0.9997

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 350

**Error Propagation:**
- Mean Error Depth: 1.7000
- Max Error Depth: 1.7000
- Error Rate: 0.4096

**Failed Node Impact:**
- Avg Degree Centrality: 0.4074
- Avg Betweenness Centrality: 0.3333
- Failure Rate: 0.3000

### cascade_topology_gat

**Performance Metrics:**
- Final Accuracy: 0.6368
- Average Accuracy: 0.6240
- Accuracy Std Dev: 0.0198
- Robustness Score: 0.9996

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 23
- Network Density: 0.2556

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 442

**Error Propagation:**
- Mean Error Depth: 2.7000
- Max Error Depth: 4.5000
- Error Rate: 0.3825

**Failed Node Impact:**
- Avg Degree Centrality: 0.3704
- Avg Betweenness Centrality: 0.2222
- Failure Rate: 0.3000

### feedback_rewired_topology_gat

**Performance Metrics:**
- Final Accuracy: 0.5808
- Average Accuracy: 0.5841
- Accuracy Std Dev: 0.0172
- Robustness Score: 0.9886

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 15
- Network Density: 0.1667

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 293

**Error Propagation:**
- Mean Error Depth: 1.9802
- Max Error Depth: 2.7143
- Error Rate: 0.3862

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.1076
- Failure Rate: 0.4000

### mesh_topology_gat

**Performance Metrics:**
- Final Accuracy: 0.6420
- Average Accuracy: 0.6315
- Accuracy Std Dev: 0.0260
- Robustness Score: 0.9993

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 1770

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.3739

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.3000

### scale_free_topology_gat

**Performance Metrics:**
- Final Accuracy: 0.5975
- Average Accuracy: 0.6017
- Accuracy Std Dev: 0.0134
- Robustness Score: 0.9998

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 32
- Network Density: 0.3556

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 620

**Error Propagation:**
- Mean Error Depth: 1.5500
- Max Error Depth: 1.9000
- Error Rate: 0.3985

**Failed Node Impact:**
- Avg Degree Centrality: 0.3333
- Avg Betweenness Centrality: 0.0463
- Failure Rate: 0.2000

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: mesh_topology_gat achieved 0.642 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: scale_free_topology_gat achieved 1.000 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. 💪 **Failure Resilience**: star_topology_gat maintains performance best under node failures. This topology has effective redundancy and graceful degradation.

5. ⚡ **Communication Efficient**: cascade_topology_gat achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **mesh_topology_gat** topology achieved the highest final accuracy of 0.6420. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*