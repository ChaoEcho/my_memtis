import numpy as np
import networkx as nx

def pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6):
    nodes = list(graph.nodes())
    n = len(nodes)
    adj_matrix = nx.to_numpy_array(graph)
    row_sums = adj_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    adj_matrix = adj_matrix / row_sums[:, np.newaxis]
    
    rank = np.ones(n) / n
    teleport = np.ones(n) / n
    
    for _ in range(max_iter):
        new_rank = alpha * adj_matrix.T @ rank + (1 - alpha) * teleport
        if np.linalg.norm(new_rank - rank, ord=1) < tol:
            break
        rank = new_rank
    
    return dict(zip(nodes, rank))

# 构建一个简单的有向图
G = nx.DiGraph()
G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])

# 运行PageRank
ranks = pagerank(G)
print("PageRank Scores:", ranks)

