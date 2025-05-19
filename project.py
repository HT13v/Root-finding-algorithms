import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

# Newton-Raphson Method
def newton_raphson(x0, tol=1e-5, max_iter=10):
    steps = [x0]
    for _ in range(max_iter):
        x1 = x0 - f(x0)/df(x0)
        steps.append(x1)
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return steps

# Bisection Method
def bisection(a, b, tol=1e-5, max_iter=50):
    steps = []
    for _ in range(max_iter):
        c = (a + b) / 2.0
        steps.append(c)
        if abs(f(c)) < tol:
            break
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return steps

# False Position (Regula Falsi)
def false_position(a, b, tol=1e-5, max_iter=50):
    steps = []
    for _ in range(max_iter):
        c = (a*f(b) - b*f(a)) / (f(b) - f(a))
        steps.append(c)
        if abs(f(c)) < tol:
            break
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return steps

# Secant Method
def secant(x0, x1, tol=1e-5, max_iter=20):
    steps = [x0, x1]
    for _ in range(max_iter):
        if f(x1) - f(x0) == 0:
            break
        x2 = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
        steps.append(x2)
        if abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return steps

# Plotting function
def plot_method(steps, method_name, color):
    x_vals = np.linspace(-3, 3, 400)
    y_vals = f(x_vals)
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.scatter(steps, [f(x) for x in steps], color=color, label=method_name)
    plt.plot(steps, [f(x) for x in steps], '--', color=color)
    plt.title(f"{method_name} Method")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run and visualize all methods
plot_method(newton_raphson(1.5), "Newton-Raphson", "blue")
plot_method(bisection(1, 2), "Bisection", "red")
plot_method(false_position(1, 2), "False Position", "green")
plot_method(secant(1, 2), "Secant", "purple")
