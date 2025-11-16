# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-11-16 17:19:06

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
| star_topology_rf | 0.4985 | 0.4947 | 0.9999 | 1.6000 | 0.2000 |
| cascade_topology_rf | 0.4889 | 0.4971 | 0.9995 | 2.6600 | 0.2444 |
| feedback_rewired_topology_rf | 0.5025 | 0.4910 | 0.9981 | 3.3000 | 0.1444 |
| mesh_topology_rf | 0.4877 | 0.4997 | 0.9994 | 0.9000 | 1.0000 |
| scale_free_topology_rf | 0.4889 | 0.5006 | 0.9998 | 1.4800 | 0.3556 |

## Detailed Analysis

### star_topology_rf

**Performance Metrics:**
- Final Accuracy: 0.4985
- Average Accuracy: 0.4947
- Accuracy Std Dev: 0.0113
- Robustness Score: 0.9999

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 352

**Error Propagation:**
- Mean Error Depth: 1.6000
- Max Error Depth: 1.7000
- Error Rate: 0.5449

**Failed Node Impact:**
- Avg Degree Centrality: 0.1111
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.2000

### cascade_topology_rf

**Performance Metrics:**
- Final Accuracy: 0.4889
- Average Accuracy: 0.4971
- Accuracy Std Dev: 0.0227
- Robustness Score: 0.9995

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 22
- Network Density: 0.2444

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 353

**Error Propagation:**
- Mean Error Depth: 2.6600
- Max Error Depth: 3.7000
- Error Rate: 0.5449

**Failed Node Impact:**
- Avg Degree Centrality: 0.2667
- Avg Betweenness Centrality: 0.2750
- Failure Rate: 0.5000

### feedback_rewired_topology_rf

**Performance Metrics:**
- Final Accuracy: 0.5025
- Average Accuracy: 0.4910
- Accuracy Std Dev: 0.0203
- Robustness Score: 0.9981

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 13
- Network Density: 0.1444

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 241

**Error Propagation:**
- Mean Error Depth: 3.3000
- Max Error Depth: 4.0000
- Error Rate: 0.5326

**Failed Node Impact:**
- Avg Degree Centrality: 0.2444
- Avg Betweenness Centrality: 0.3500
- Failure Rate: 0.5000

### mesh_topology_rf

**Performance Metrics:**
- Final Accuracy: 0.4877
- Average Accuracy: 0.4997
- Accuracy Std Dev: 0.0241
- Robustness Score: 0.9994

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 1520

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.5018

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.6000

### scale_free_topology_rf

**Performance Metrics:**
- Final Accuracy: 0.4889
- Average Accuracy: 0.5006
- Accuracy Std Dev: 0.0142
- Robustness Score: 0.9998

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 32
- Network Density: 0.3556

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 556

**Error Propagation:**
- Mean Error Depth: 1.4800
- Max Error Depth: 1.9000
- Error Rate: 0.5449

**Failed Node Impact:**
- Avg Degree Centrality: 0.2667
- Avg Betweenness Centrality: 0.0231
- Failure Rate: 0.5000

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: feedback_rewired_topology_rf achieved 0.502 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: star_topology_rf achieved 1.000 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. 💪 **Failure Resilience**: mesh_topology_rf maintains performance best under node failures. This topology has effective redundancy and graceful degradation.

5. ⚡ **Communication Efficient**: star_topology_rf, feedback_rewired_topology_rf achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **feedback_rewired_topology_rf** topology achieved the highest final accuracy of 0.5025. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*