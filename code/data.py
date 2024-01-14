import numpy as np
# Set the learning rate
learning_rate = 0.01

# Redefine the number of iterations
num_iterations = 2000
adj_matrix = np.array([
    [0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0],
    [0, 0, 0, 0.33333333, 0.33333333, 0.33333333, 0, 0, 0, 0],
    [0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],
    [0, 0.2, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0],
    [0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0],
    [0.33333333, 0.33333333, 0, 0.33333333, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0],
    [0, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0],
    [0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.5],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
])

# Reinitialize node values
node_values = np.arange(10, dtype=float)

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

# Display the final standard deviation, and sample values from the convergence_rates list
final_std_dev = std_devs[-1]
sample_convergence_rates = convergence_rates[::100]  # Sample every 100 iterations
print(std_devs)
print(convergence_rates)

'''
import matplotlib.pyplot as plt

# Plotting the standard deviation over iterations
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(std_devs, label="Standard Deviation")
plt.xlabel("Iteration")
plt.ylabel("Standard Deviation")
plt.title("Standard Deviation over Iterations")
plt.grid(True)
plt.legend()

# Plotting the convergence rate over iterations
plt.subplot(1, 2, 2)
plt.plot(convergence_rates, label="Convergence Rate", color='orange')
plt.xlabel("Iteration")
plt.ylabel("Convergence Rate")
plt.title("Convergence Rate over Iterations")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
'''