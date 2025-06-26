import numpy as np

def generate_sample_points(num_points, x_min, x_max):
    # Generate num_points random values between x_min and x_max
    sample_points = np.random.uniform(x_min, x_max, num_points)
    return sample_points




import numpy as np
import matplotlib.pyplot as plt

def test_function(x):
    return np.sin(x) + np.sin(10 * x / 3)

# Generate a set of evenly spaced points within the range [-5, 5]
x_values = np.linspace(-5, 5, 400)
y_values = test_function(x_values)

# Create a plot to visualize the test function
plt.plot(x_values, y_values, label='f(x) = sin(x) + sin(10*x/3)')
plt.title('Plot of the Multi-modal Test Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()