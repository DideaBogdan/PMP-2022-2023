import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()


with model:
    alpha = 4.7
    traffic = pm.Poisson("T", mu=20)
    order = pm.Normal("O", mu=1, sigma=0.5)
    cook = pm.Exponential("C", lam=alpha)
    trace = pm.sample(500, chains=1)

dictionary = {
    "T_C": trace["C"].tolist(),
    "O" : trace["O"].tolist()
}
df = pd.DataFrame(dictionary)

wait_time = df[(df["O"] + df["T_C"] <= 15)]
print("Timpul de asteptare <= 15:", wait_time.shape[0] / df.shape[0])

total = 0
for tup in zip(df["O"], df["T_C"]):
    total += tup[0] + tup[1]

print("Timpul mediu de asteptare:", total / df.shape[0])

az.plot_posterior({"Clienti pe ora": trace["T"], "Timp de asteptare": trace["O"],
                   "Medie comenzi": trace["C"]})
plt.show()
