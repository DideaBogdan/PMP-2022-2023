import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()
alpha = 0.5

with model:
    traffic = pm.Poisson("T", mu=20)
    order = pm.Normal("O", 1, sigma=0.5)
    cook = pm.Exponential("C", lam=alpha)
    trace = pm.sample(2000, chains=1)


az.plot_posterior(trace)
plt.show()
