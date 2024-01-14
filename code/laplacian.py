import numpy as np
import ast
import networkx as nx
import matplotlib.pyplot as plt


def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read the content of the file
        data_str = file.read()
        # Convert string representation of list to actual list
        data_list = ast.literal_eval(data_str)
    return data_list


# Path to the file for powerlaw data
file_powerlaw = "/Users/jichanglong/Desktop/TopologyInfluence/graphGeneration/powerlaw_normal.txt"
data_powerlaw = read_data_from_file(file_powerlaw)

# Path to the file for poisson data
file_poisson = "/Users/jichanglong/Desktop/TopologyInfluence/graphGeneration/poisson_normal.txt"
data_poisson = read_data_from_file(file_poisson)


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
power_law_fiedler = calculate_fiedler_values(data_powerlaw)
poisson_fiedler = calculate_fiedler_values(data_poisson)

print(power_law_fiedler)
print(poisson_fiedler)

# Create a plot to compare the Fiedler values
plt.figure(figsize=(10, 6))
x_values = range(1, len(power_law_fiedler) + 1)  # x坐标从1开始

plt.plot(x_values, power_law_fiedler, label="Powerlaw")
plt.plot(x_values, poisson_fiedler, label="Poisson")
plt.xlabel("x-th graph")
plt.ylabel("Fiedler Value")
plt.title("Comparison of Fiedler Values")
plt.grid(True)
plt.legend()
plt.show()

