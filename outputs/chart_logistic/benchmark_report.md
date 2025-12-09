# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-12-08 17:26:33

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
- **Topologies Tested**: star_topology_logistic, cascade_topology_logistic, feedback_rewired_topology_logistic, mesh_topology_logistic, scale_free_topology_logistic

## Results Summary

| Topology | Final Accuracy | Avg Accuracy | Robustness | Error Depth | Network Density |
|----------|----------------|--------------|------------|-------------|-----------------|
| star_topology_logistic | 0.6988 | 0.6755 | 0.9998 | 1.7000 | 0.2000 |
| cascade_topology_logistic | 0.6667 | 0.6598 | 0.9981 | 2.7778 | 0.2667 |
| feedback_rewired_topology_logistic | 0.6808 | 0.6442 | 0.9922 | 1.7429 | 0.1556 |
| mesh_topology_logistic | 0.6836 | 0.6712 | 0.9998 | 0.9000 | 1.0000 |
| scale_free_topology_logistic | 0.6653 | 0.6719 | 0.9996 | 1.5556 | 0.3556 |

## Detailed Analysis

### star_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.6988
- Average Accuracy: 0.6755
- Accuracy Std Dev: 0.0144
- Robustness Score: 0.9998

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 300

**Error Propagation:**
- Mean Error Depth: 1.7000
- Max Error Depth: 1.7000
- Error Rate: 0.3198

**Failed Node Impact:**
- Avg Degree Centrality: 0.2889
- Avg Betweenness Centrality: 0.2000
- Failure Rate: 0.5000

### cascade_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.6667
- Average Accuracy: 0.6598
- Accuracy Std Dev: 0.0215
- Robustness Score: 0.9981

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 24
- Network Density: 0.2667

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 398

**Error Propagation:**
- Mean Error Depth: 2.7778
- Max Error Depth: 4.5000
- Error Rate: 0.3370

**Failed Node Impact:**
- Avg Degree Centrality: 0.3333
- Avg Betweenness Centrality: 0.1331
- Failure Rate: 0.1000

### feedback_rewired_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.6808
- Average Accuracy: 0.6442
- Accuracy Std Dev: 0.0196
- Robustness Score: 0.9922

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 14
- Network Density: 0.1556

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 238

**Error Propagation:**
- Mean Error Depth: 1.7429
- Max Error Depth: 2.7000
- Error Rate: 0.3444

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.0231
- Failure Rate: 0.3000

### mesh_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.6836
- Average Accuracy: 0.6712
- Accuracy Std Dev: 0.0144
- Robustness Score: 0.9998

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 1618

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.3284

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.2000

### scale_free_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.6653
- Average Accuracy: 0.6719
- Accuracy Std Dev: 0.0194
- Robustness Score: 0.9996

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 32
- Network Density: 0.3556

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 646

**Error Propagation:**
- Mean Error Depth: 1.5556
- Max Error Depth: 1.9000
- Error Rate: 0.3260

**Failed Node Impact:**
- Avg Degree Centrality: 0.4444
- Avg Betweenness Centrality: 0.1157
- Failure Rate: 0.1000

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: star_topology_logistic achieved 0.699 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: mesh_topology_logistic achieved 1.000 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. 💪 **Failure Resilience**: scale_free_topology_logistic maintains performance best under node failures. This topology has effective redundancy and graceful degradation.

5. ⚡ **Communication Efficient**: star_topology_logistic, feedback_rewired_topology_logistic achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **star_topology_logistic** topology achieved the highest final accuracy of 0.6988. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*