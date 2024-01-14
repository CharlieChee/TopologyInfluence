import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
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
for i in range(5000):
     total.append(create_normalized_adj_matrix(30, 60))

print("generated")


total_std = []
def write_to_files():
    for matrix in total:
        learning_rate = 0.1

        # Redefine the number of iterations
        num_iterations = 100
        adj_matrix = matrix
        for i in range(len(matrix)):
            if sum(matrix[i]) != 1:
                matrix[i] = matrix[i] / sum(matrix[i])

        # Reinitialize node values
        node_values = np.arange(len(matrix), dtype=float)

        # Initialize arrays to store the standard deviation and convergence rate after each iteration
        std_devs = []
        convergence_rates = []

        for iteration in range(num_iterations):
            new_values = np.copy(node_values)
            for i in range(len(node_values)):
                # Get the neighbors and their probabilities
                neighbors = np.nonzero(adj_matrix[i])[0]
                probabilities = adj_matrix[i][neighbors]

                if len(neighbors) > 0:
                    # Choose a neighbor based on the defined probabilities
                    chosen_neighbor = np.random.choice(neighbors, p=probabilities)

                    # Calculate the value update based on the average value and learning rate
                    average_value = (node_values[i] + node_values[chosen_neighbor]) / 2
                    new_values[i] += learning_rate * (average_value - node_values[i])

            # Update node values after all nodes have chosen their neighbors for this iteration
            node_values = new_values

            # Calculate and store the current standard deviation
            current_std_dev = np.std(node_values)
            std_devs.append(current_std_dev)
        total_std.append(std_devs)
    return total_std

a1 = write_to_files()
speed = []
for i in a1:
    speed.append(1/np.mean(i))

print("speed done")
def calculate_fiedler_values(data_list):
    fiedler_values = []
    for adj_matrix in data_list:
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian_matrix = degree_matrix - adj_matrix

        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(laplacian_matrix)

        # Find the second smallest non-zero eigenvalue (Fiedler value)
        fiedler_value = sorted(eigenvalues)[1]

        fiedler_values.append(fiedler_value)

    return fiedler_values


# Calculate Fiedler values for powerlaw and poisson data
fiedler = calculate_fiedler_values(total)
print("fiedler done")

print(len(fiedler),len(speed))

plt.scatter(fiedler,speed, marker='o', color='b', label='Data Points')

# 添加图表标题和坐标轴标签
plt.title('Scatter Plot of 5000 Matrices')
plt.xlabel('Fiedler value')
plt.ylabel('Convergence speed')

# 显示图例
plt.legend()

# 显示图像
plt.show()