import networkx as nx
import numpy as np
# Function to create a connected graph and return its normalized adjacency matrix
def create_normalized_adj_matrix(num_nodes, num_edges):
    # Create a connected graph
    while True:
        G = nx.gnm_random_graph(num_nodes, num_edges)
        if nx.is_connected(G):
            break

    # Create the adjacency matrix
    adj_matrix = nx.to_numpy_array(G)

    # Normalize the adjacency matrix
    # Avoid division by zero for nodes with no neighbors
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = adj_matrix.sum(axis=1)
        normalized_adj_matrix = np.divide(adj_matrix, row_sums[:, np.newaxis], out=np.zeros_like(adj_matrix), where=row_sums[:, np.newaxis] != 0)

    return normalized_adj_matrix
total = []
for i in range(100):
     total.append(create_normalized_adj_matrix(20, 30))

print(total)