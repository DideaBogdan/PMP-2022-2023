import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az


#Exercitiul 1
x = stats.expon.rvs(scale = 1 / 4, size = 10000)
y = stats.expon.rvs(scale = 1 / 6, size = 10000)
z = stats.binom.rvs(1, 0.4, size = 10000)

q = []
for i in range(1, 10000):
    if z[i] == 1:
        q.append(x[i])
    else:
        q.append(y[i])


az.plot_posterior({'q':q}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show()
print("media este: ", np.mean(q))
print("deviatia standard este ", np.std(q))


#Exercitiul 2

server1 = stats.gamma.rvs(4, scale = 1/4, size = 10000)
server2 = stats.gamma.rvs(4, scale = 1/3, size = 10000)
server3 = stats.gamma.rvs(5, scale = 1/2, size = 10000)
server4 = stats.gamma.rvs(5, scale = 1/3, size = 10000)
servertime = stats.expon.rvs(scale = 1/4, size = 10000)

direct_to_server = stats.binom.rvs(4, 0.25, size = 10000)
responsetime = []

for i in range(1, 10000):
    if direct_to_server[i] == 0:
        responsetime.append(server1[i])
    elif direct_to_server[i] == 1:
        responsetime.append(server2[i])
    elif direct_to_server[i] == 2:
        responsetime.append(server3[i])
    elif direct_to_server[i] == 3:
        responsetime.append(server4[i])

more_than_3 = 0
for index in responsetime:
    if index+servertime >=3:
        more_than_3 +=1

print("probabilitatea sa dureze mai mult de 3 milisecunde este de ", more_than_3/10000)

az.plot_posterior({'response time':responsetime}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show()
print("media este: ", np.mean(responsetime))
print("deviatia standard este ", np.std(responsetime))
