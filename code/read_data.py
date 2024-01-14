import numpy as np
import matplotlib.pyplot as plt

def read_from_files(file_name):
    result = []
    with open(file_name, "r") as file:
        for line in file:
            # 去除字符串两端的空白字符，并将其解析为列表
            result.append(eval(line.strip()))
    return result

# Read data for std and convergence rate
a1_std = read_from_files("/Users/jichanglong/Desktop/pythonProject1/powerlaw_normal_std.txt")
a2_std = read_from_files("/Users/jichanglong/Desktop/pythonProject1/poisson_normal_std.txt")
a1_convergence_rate = read_from_files("/Users/jichanglong/Desktop/pythonProject1/powerlaw_normal_convergence_rate.txt")
a2_convergence_rate = read_from_files("/Users/jichanglong/Desktop/pythonProject1/poisson_normal_convergence_rate.txt")

# Calculate averages for std and convergence rate
std_devs_a1 = [sum(column) / len(column) for column in zip(*a1_std)]
std_devs_a2 = [sum(column) / len(column) for column in zip(*a2_std)]
convergence_rates_a1 = [sum(column) / len(column) for column in zip(*a1_convergence_rate)]
convergence_rates_a2 = [sum(column) / len(column) for column in zip(*a2_convergence_rate)]

# Create a figure with two subplots for std and convergence rate
plt.figure(figsize=(12, 6))

# Plotting the standard deviation over iterations
plt.subplot(1, 2, 1)
plt.plot(std_devs_a1, label="Powerlaw Normal")
plt.plot(std_devs_a2, label="Poisson Normal")
plt.xlabel("Iteration")
plt.ylabel("Standard Deviation")
plt.title("Standard Deviation over Iterations")
plt.grid(True)
plt.legend()

# Plotting the convergence rate over iterations
plt.subplot(1, 2, 2)
plt.plot(convergence_rates_a1, label="Powerlaw Normal", color='orange')
plt.plot(convergence_rates_a2, label="Poisson Normal", color='green')
plt.xlabel("Iteration")
plt.ylabel("Convergence Rate")
plt.title("Convergence Rate over Iterations")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
