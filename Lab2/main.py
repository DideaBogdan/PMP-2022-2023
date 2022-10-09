import random

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

# Exercitiul 1
x = stats.expon.rvs(scale=1 / 4, size=10000)
y = stats.expon.rvs(scale=1 / 6, size=10000)
z = stats.binom.rvs(1, 0.4, size=10000)

q = []
for i in range(1, 10000):
    if z[i] == 1:
        q.append(x[i])
    else:
        q.append(y[i])

az.plot_posterior({'q': q})  # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show()
print("media este: ", np.mean(q))
print("deviatia standard este ", np.std(q))

# Exercitiul 2

server1 = stats.gamma.rvs(4, scale=1 / 3, size=10000)
server2 = stats.gamma.rvs(4, scale=1 / 2, size=10000)
server3 = stats.gamma.rvs(5, scale=1 / 2, size=10000)
server4 = stats.gamma.rvs(5, scale=1 / 3, size=10000)
servertime = stats.expon.rvs(scale=1 / 4, size=10000)

direct_to_server = []
direct_to_server = random.choices((0, 1, 2, 3), weights=(0.25, 0.25, 0.30, 0.20), k=10000)

response_time = []

for i in range(1, 10000):
    if direct_to_server[i] == 0:
        response_time.append(server1[i] + direct_to_server[i])
    elif direct_to_server[i] == 1:
        response_time.append(server2[i])
    elif direct_to_server[i] == 2:
        response_time.append(server3[i])
    elif direct_to_server[i] == 3:
        response_time.append(server4[i])

more_than_3 = 0
for i in range(len(response_time)):
    if response_time[i] >= 3:
        more_than_3 += 1

print("probabilitatea sa dureze mai mult de 3 milisecunde este de ", more_than_3 / 10000)

az.plot_posterior({'response time': response_time})  # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show()
print("media este: ", np.mean(response_time))
print("deviatia standard este ", np.std(response_time))

# Exercise 3

n = 10 #se presupune ca numarul generat este stema
fair_coin = stats.binom.rvs(n, 0.5, size=100)
biased_coin = stats.binom.rvs(n, 0.3, size=100)

ss = []
sb = []
bs = []
bb = []
for i in range(0, 100):
    ss.append((fair_coin[i] + biased_coin[i])/20)
    sb.append((fair_coin[i] + (10 - biased_coin[i])) / 20)
    bs.append(((10 - fair_coin[i]) + biased_coin[i]) / 20)
    bb.append((20 - (fair_coin[i] + biased_coin[i])) / 20)

az.plot_posterior({'ss': ss, 'sb': sb, 'bs': bs, 'bb': bb})  # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show()
