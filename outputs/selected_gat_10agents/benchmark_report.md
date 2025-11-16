# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-11-16 17:15:21

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
| star_topology_gat | 0.6123 | 0.6233 | 0.9995 | 1.5400 | 0.2000 |
| cascade_topology_gat | 0.6513 | 0.6360 | 0.9995 | 3.2000 | 0.2111 |
| feedback_rewired_topology_gat | 0.5895 | 0.6003 | 0.9998 | 2.1667 | 0.1333 |
| mesh_topology_gat | 0.6728 | 0.6603 | 0.9994 | 0.9000 | 1.0000 |
| scale_free_topology_gat | 0.6399 | 0.6434 | 0.9995 | 1.5667 | 0.3556 |

## Detailed Analysis

### star_topology_gat

**Performance Metrics:**
- Final Accuracy: 0.6123
- Average Accuracy: 0.6233
- Accuracy Std Dev: 0.0231
- Robustness Score: 0.9995

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 430

**Error Propagation:**
- Mean Error Depth: 1.5400
- Max Error Depth: 1.7000
- Error Rate: 0.3702

**Failed Node Impact:**
- Avg Degree Centrality: 0.1111
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.5000

### cascade_topology_gat

**Performance Metrics:**
- Final Accuracy: 0.6513
- Average Accuracy: 0.6360
- Accuracy Std Dev: 0.0225
- Robustness Score: 0.9995

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 19
- Network Density: 0.2111

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 363

**Error Propagation:**
- Mean Error Depth: 3.2000
- Max Error Depth: 4.5000
- Error Rate: 0.3493

**Failed Node Impact:**
- Avg Degree Centrality: 0.2778
- Avg Betweenness Centrality: 0.3889
- Failure Rate: 0.2000

### feedback_rewired_topology_gat

**Performance Metrics:**
- Final Accuracy: 0.5895
- Average Accuracy: 0.6003
- Accuracy Std Dev: 0.0135
- Robustness Score: 0.9998

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 12
- Network Density: 0.1333

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 248

**Error Propagation:**
- Mean Error Depth: 2.1667
- Max Error Depth: 2.5000
- Error Rate: 0.3875

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.1088
- Failure Rate: 0.6000

### mesh_topology_gat

**Performance Metrics:**
- Final Accuracy: 0.6728
- Average Accuracy: 0.6603
- Accuracy Std Dev: 0.0251
- Robustness Score: 0.9994

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 1978

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.3383

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.2000

### scale_free_topology_gat

**Performance Metrics:**
- Final Accuracy: 0.6399
- Average Accuracy: 0.6434
- Accuracy Std Dev: 0.0223
- Robustness Score: 0.9995

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 32
- Network Density: 0.3556

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 616

**Error Propagation:**
- Mean Error Depth: 1.5667
- Max Error Depth: 1.9000
- Error Rate: 0.3542

**Failed Node Impact:**
- Avg Degree Centrality: 0.3611
- Avg Betweenness Centrality: 0.1007
- Failure Rate: 0.4000

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: mesh_topology_gat achieved 0.673 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: feedback_rewired_topology_gat achieved 1.000 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. 💪 **Failure Resilience**: feedback_rewired_topology_gat maintains performance best under node failures. This topology has effective redundancy and graceful degradation.

5. ⚡ **Communication Efficient**: cascade_topology_gat, scale_free_topology_gat achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **mesh_topology_gat** topology achieved the highest final accuracy of 0.6728. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*