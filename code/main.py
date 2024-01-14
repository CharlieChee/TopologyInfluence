file = "/Users/jichanglong/Desktop/TopologyInfluence/graphGeneration/powerlaw.txt"
import numpy as np
import ast
import networkx as nx
def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read the content of the file
        data_str = file.read()
        # Convert string representation of list to actual list
        data_list = ast.literal_eval(data_str)
    return data_list

# Path to the file

file2 = "/Users/jichanglong/Desktop/TopologyInfluence/graphGeneration/poisson.txt"
# Read and convert the data
data_list = read_data_from_file(file2)
total = []
for i in data_list:
    total.append(np.array(i))
#print(total)


'''
#for normalization
matrix_normal = []
for matrix in total:
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_normal.append(matrix / row_sums)
print(matrix_normal)
'''
'''
import numpy as np


def sinkhorn_knopp(matrix, tolerance=1e-10, max_iterations=1000):
    """
    Apply the Sinkhorn-Knopp algorithm to make a matrix doubly stochastic.

    :param matrix: A 2D numpy array with non-negative entries.
    :param tolerance: Convergence tolerance.
    :param max_iterations: Maximum number of iterations.
    :return: Doubly stochastic matrix.
    """
    if not np.all(matrix >= 0):
        raise ValueError("Matrix entries must be non-negative")

    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)

    if np.any(row_sums == 0) or np.any(col_sums == 0):
        raise ValueError("Matrix contains zero rows/columns")

    # Normalize the matrix
    matrix = matrix / row_sums[:, np.newaxis]

    for _ in range(max_iterations):
        # Scale columns
        matrix = matrix / matrix.sum(axis=0)

        # Scale rows
        matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]

        # Check for convergence
        if np.max(np.abs(matrix.sum(axis=1) - 1)) < tolerance and \
                np.max(np.abs(matrix.sum(axis=0) - 1)) < tolerance:
            return matrix

    return matrix
matrix_doubly = []

for i in total:
    matrix_doubly.append(np.array(sinkhorn_knopp(i)))
print(matrix_doubly)

'''