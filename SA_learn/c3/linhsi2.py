import numpy as np
import matplotlib.pyplot as plt

def test_function(x):
    return np.sin(x) + np.sin(10 * x / 3)

# Generate a neighboring solution using a more adaptive method
def generate_neighbor(solution, step_size, min_max, temperature):
    # Step size decreases as temperature decreases (adapting to cooling schedule)
    adaptive_step = step_size * temperature
    perturbation = np.random.uniform(-adaptive_step, adaptive_step)
    candidate = solution + perturbation
    # limit new solutions to the search space
    candidate = np.clip(candidate, min_max[0], min_max[1])
    return candidate

# Calculate acceptance probability
def acceptance_probability(current_cost, new_cost, temperature):
    if new_cost < current_cost:
        return 1.0
    else:
        return np.exp(-(new_cost - current_cost) / (1e-8 + temperature))

# Simulated Annealing main function
def simulated_annealing(initial_solution, objective_function, min_max, T_start, alpha, num_iterations, step_size):
    current_solution = initial_solution
    current_cost = objective_function(current_solution)
    best_solution = current_solution
    best_cost = current_cost
    accepted_solutions = [current_solution]
    costs = [current_cost]

    temperature = T_start
    for iteration in range(num_iterations):
        # Generate neighboring solution using adaptive step size based on temperature
        neighbor = generate_neighbor(current_solution, step_size, min_max, temperature)
        neighbor_cost = objective_function(neighbor)
        if acceptance_probability(current_cost, neighbor_cost, temperature) > np.random.random():
            current_solution = neighbor
            current_cost = neighbor_cost
            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost

        accepted_solutions.append(current_solution)
        costs.append(current_cost)
        temperature *= alpha

    return accepted_solutions, best_solution, best_cost, costs

# Run the algorithm
T_start = 100
alpha = 0.95
num_iterations = 1000
step_size = 0.5
min_max = [-5, 5]
initial_solution = np.random.uniform(min_max[0], min_max[1])
results = simulated_annealing(initial_solution, test_function, min_max, T_start, alpha, num_iterations, step_size)

# Plot the results
accepted_solutions, best_solution, best_cost, costs = results

plt.figure(figsize=(12, 6))

# Plot the cost
plt.subplot(1, 2, 1)
plt.plot(costs, label='Cost')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost by Iteration')

# Plot the solution
plt.subplot(1, 2, 2)
plt.plot(accepted_solutions, label='Solution', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Solution')
plt.title('Solution by Iteration')

# Add text annotations for the best solution and cost
plt.subplot(1, 2, 2)
plt.annotate(f'Best solution: {best_solution:.4f}\nCost: {best_cost:.4f}',
             xy=(num_iterations-1, best_solution),
             xycoords='data',
             xytext=(-50, 20),
             textcoords='offset points',
             arrowprops=dict(arrowstyle="->", lw=1),
             fontsize=12, color='red')

plt.tight_layout()
plt.show()

# Output the best solution found
best_solution, best_cost
