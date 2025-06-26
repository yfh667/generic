import matplotlib.pyplot as plt
import math
import numpy as np

def linear_cooling(T_start, alpha, iteration):
    """ Calculate the temperature using a linear cooling schedule. """
    return T_start - alpha * iteration

def exponential_cooling(T_start, alpha, iteration):
    """ Calculate the temperature using an exponential cooling schedule. """
    return T_start * (alpha ** iteration)

def logarithmic_cooling(C, iteration):
    """ Calculate the temperature using a logarithmic cooling schedule. """
    return C / (1e-8 + math.log(1 + iteration))

def generate_sample_points(num_points, x_min, x_max, temperature):
    """ Generate sample points based on the current temperature. """
    # Calculate the midpoint of the range
    midpoint = (x_min + x_max) / 2
    # Generate points from a normal distribution centered at midpoint with std dev proportional to temperature
    sample_points = np.random.normal(midpoint, temperature, num_points)
    return sample_points

def visualize_cooling_schedules(num_iterations, num_points, x_min, x_max, T_start, alpha, C):
    """ Visualize the effect of different cooling schedules on point distributions. """
    plt.figure(figsize=(15, 5))

    # Prepare subplots for each cooling schedule
    schedules = ['Linear', 'Exponential', 'Logarithmic']
    for i, schedule in enumerate(schedules, 1):
        plt.subplot(1, 3, i)
        for iteration in range(num_iterations):
            if schedule == 'Linear':
                temperature = linear_cooling(T_start, alpha, iteration)
            elif schedule == 'Exponential':
                temperature = exponential_cooling(T_start, alpha, iteration)
            else:  # Logarithmic
                temperature = logarithmic_cooling(C, iteration)

            points = generate_sample_points(num_points, x_min, x_max, temperature)
            plt.scatter([iteration] * num_points, points, alpha=0.6, s=10)

        plt.title(f'{schedule} Cooling Schedule')
        plt.xlabel('Iteration')
        plt.ylabel('Point Distribution')

    plt.tight_layout()
    plt.show()

# Example usage
visualize_cooling_schedules(50, 100, -5, 5, 10, 0.1, 20)