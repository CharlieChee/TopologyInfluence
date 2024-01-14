import numpy as np
import ast
import networkx as nx
# Function to read data from a file and convert to a list
def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read the content of the file
        data_str = file.read()
        # Convert string representation of list to actual list
        data_list = ast.literal_eval(data_str)
    return data_list

# Path to the file

file2 = "/Users/jichanglong/Desktop/TopologyInfluence/graphGeneration/random.txt"
# Read and convert the data
data_list = read_data_from_file(file2)
total = []
for i in data_list:
    total.append(np.array(i))

def write_to_files():
    for matrix in total:
        learning_rate = 0.01

        # Redefine the number of iterations
        num_iterations = 2000
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

            # Calculate and store the convergence rate (except for the first iteration)
            if iteration > 0:
                convergence_rate = abs(current_std_dev - std_devs[iteration - 1]) / std_devs[iteration - 1]
                convergence_rates.append(convergence_rate)

        a1 = std_devs
        a2 = convergence_rates

        with open("a1.txt", "a") as file1:
            file1.write(str(a1) + "\n")

        with open("a2.txt", "a") as file2:
            file2.write(str(a2) + "\n")


write_to_files()