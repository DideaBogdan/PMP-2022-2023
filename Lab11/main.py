import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics

#Exercitiul 1
def posterior_grid1(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points) # uniform prior
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid1(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()


def posterior_grid2(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    prior = (grid<= 0.5).astype(int)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


data = np.repeat([0, 1], (50, 50))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid2(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()


def posterior_grid3(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


data = np.repeat([0, 1], (115, 40))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid3(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()


#Exercitiul 2
def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    outside = np.invert(inside)
    plt.figure(figsize=(8, 8))
    plt.plot(x[inside], y[inside], 'b.')
    plt.plot(x[outside], y[outside], 'r.')
    plt.plot(0, 0, label=f'π*= {pi:4.3f}\nerror = {error:4.3f}', alpha=0)
    plt.axis('square')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc=1, frameon=True, framealpha=0.9)
    plt.show()
    return error

error_list = []
error_list.append(estimate_pi(100))
error_list.append(estimate_pi(1000))
error_list.append(estimate_pi(10000))

error_mean = statistics.mean(error_list)
error_std = statistics.stdev(error_list)
plt.errorbar(-1, 1, yerr=error_mean, xerr=error_std)
plt.show()

#Exercitiul 3

def metropolis(func, draws=10000):
    """A very simple Metropolis implementation"""
    trace = np.zeros(draws)
    old_x = 0.5  # func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace


beta_params = [(1, 1), (20, 20), (1, 4)]
for i,j in beta_params:
    func = stats.beta(i,j)
    trace = metropolis(func=func)
    x = np.linspace(0.01, .99, 100)
    y = func.pdf(x)
    plt.xlim(0, 1)
    plt.plot(x, y, 'C1-', lw=3, label='True distribution')
    plt.hist(trace[trace > 0], bins=25, density=True, label='Estimated distribution')
    plt.xlabel('x')
    plt.ylabel('pdf(x)')
    plt.yticks([])
    plt.legend()
    plt.show()


n_trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
data = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48]
theta_real = 0.35
beta_params = [(1, 1), (20, 20), (1, 4)]
dist = stats.beta
for idx, N in enumerate(n_trials):
    y = data[idx]
    for (a_prior, b_prior) in beta_params:
        param1, param2 = (a_prior + y, b_prior + N - y)
        func = stats.beta(param1, param2)
        trace = metropolis(func=func)
        x = np.linspace(0.01, .99, 100)
        y = func.pdf(x)
        plt.xlim(0, 1)
        plt.plot(x, y, 'C1-', lw=3, label='True distribution')
        plt.hist(trace[trace > 0], bins=25, density=True, label='Estimated distribution')
        plt.xlabel('x')
        plt.ylabel('pdf(x)')
        plt.yticks([])
        plt.legend()
        plt.show()

