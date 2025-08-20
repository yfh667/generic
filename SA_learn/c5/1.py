import numpy as np
import matplotlib.pyplot as plt

def test_function(x):
    return np.sin(x) + np.sin(10 * x / 3)

# Generate a neighboring solution
def generate_neighbor(solution, step_size, min_max):
    perturbation = np.random.uniform(-step_size, step_size)
    candidate = solution + perturbation
    # limit new solutions to the search space
    candidate = np.clip(candidate, min_max[0], min_max[1])
    return candidate

# Calculate acceptance probability
def acceptance_probability(current_cost, new_cost, temperature):
    if new_cost < current_cost:
        return 1.0
    else:
        return np.exp(-(new_cost - current_cost) / (1e-8+temperature))

def calculate_convergence_rate(costs, interval=100):
    improvements = []
    for i in range(interval, len(costs), interval):
        improvement = costs[i - interval] - costs[i]
        improvements.append(improvement)
    if improvements:
        return sum(improvements) / len(improvements)
    return 0

# Simulated Annealing main.py function
def simulated_annealing(initial_solution, objective_function, min_max, T_start, alpha, num_iterations, step_size):
    current_solution = initial_solution
    current_cost = objective_function(current_solution)
    best_solution = current_solution
    best_cost = current_cost
    accepted_solutions = [current_solution]
    costs = [current_cost]

    temperature = T_start
    for iteration in range(num_iterations):
        neighbor = generate_neighbor(current_solution, step_size, min_max)
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

def run_parameter_tuning_experiment(temperature_ranges, cooling_rates, step_sizes, objective_function, min_max, num_iterations):
    experiments = []
    for T_start in temperature_ranges:
        for alpha in cooling_rates:
            for step_size in step_sizes:
                initial_solution = np.random.uniform(min_max[0], min_max[1])
                _, best_solution, best_cost, costs = simulated_annealing(
                    initial_solution, objective_function, min_max, T_start, alpha, num_iterations, step_size
                )
                convergence_rate = calculate_convergence_rate(costs, 100)
                experiments.append({
                    "T_start": T_start,
                    "alpha": alpha,
                    "step_size": step_size,
                    "best_solution": best_solution,
                    "best_cost": best_cost,
                    "convergence_rate": convergence_rate
                })
    return experiments

def visualize_parameter_tuning_results(experiments):
    # Extract the data for plotting
    T_starts = sorted(set(exp["T_start"] for exp in experiments))
    alphas = sorted(set(exp["alpha"] for exp in experiments))
    step_sizes = sorted(set(exp["step_size"] for exp in experiments))

    # Prepare data structures for plotting
    quality_data = {T: {alpha: [] for alpha in alphas} for T in T_starts}
    convergence_data = {T: {alpha: [] for alpha in alphas} for T in T_starts}
    for exp in experiments:
        quality_data[exp["T_start"]][exp["alpha"]].append(exp["best_cost"])
        convergence_data[exp["T_start"]][exp["alpha"]].append(exp["convergence_rate"])

    # Set up the figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(12, 18))

    # Plot best solution quality
    for i, T_start in enumerate(T_starts):
        for j, alpha in enumerate(alphas):
            axs[0].plot(step_sizes, quality_data[T_start][alpha], marker='o', label=f'T={T_start}, α={alpha}')
    axs[0].set_title('Best Solution Quality by Temperature and Cooling Rate')
    axs[0].set_xlabel('Step Size')
    axs[0].set_ylabel('Best Solution Quality')
    axs[0].legend()

    # Plot convergence rate
    for i, T_start in enumerate(T_starts):
        for j, alpha in enumerate(alphas):
            axs[1].plot(step_sizes, convergence_data[T_start][alpha], marker='o', label=f'T={T_start}, α={alpha}')
    axs[1].set_title('Convergence Rate by Temperature and Cooling Rate')
    axs[1].set_xlabel('Step Size')
    axs[1].set_ylabel('Convergence Rate')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Run the algorithm
num_iterations = 1000
min_max = [-5, 5]
temperature_ranges = [1000, 100, 10]
cooling_rates = [0.99, 0.95, 0.9]
step_sizes = [5, 1, 0.1]

# Run parameter tuning experiments
experiments_results = run_parameter_tuning_experiment(temperature_ranges, cooling_rates, step_sizes, test_function, min_max, 500)

# Visualize results
visualize_parameter_tuning_results(experiments_results)