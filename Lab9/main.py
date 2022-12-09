import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az


if __name__ == '__main__':
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('date.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 2
    x_1p = np.vstack([x_1**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    with pm.Model() as model_l:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10)
        ε = pm.HalfNormal('ε', 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=2)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p2 = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
    β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
    y_l_post = α_l_post + β_l_post * x_new
    plt.plot(x_new, y_l_post, 'C1', label='linear model')
    plt.show()

    α_p_post = idata_p2.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p2.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {2}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('date.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    with pm.Model() as model_p5:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=5)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p5 = pm.sample(2000, return_inferencedata=True)

    α_p_post = idata_p5.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p5.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C3', label=f'model order {5}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    with pm.Model() as model_p5_sd_100:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=100, shape=5)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p5_sd_100 = pm.sample(2000, return_inferencedata=True)

    α_p_post = idata_p5_sd_100.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p5_sd_100.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C3', label=f'model order {5}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()
    
    
    #Pentru exrecitiul 2
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('date2.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    with pm.Model() as model_l:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10)
        ε = pm.HalfNormal('ε', 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=2)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p2 = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
    β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
    y_l_post = α_l_post + β_l_post * x_new
    plt.plot(x_new, y_l_post, 'C1', label='linear model')
    plt.show()

    α_p_post = idata_p2.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p2.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {2}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('date.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    with pm.Model() as model_p5:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=5)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p5 = pm.sample(2000, return_inferencedata=True)

    α_p_post = idata_p5.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p5.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C3', label=f'model order {5}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    with pm.Model() as model_p5_sd_100:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=100, shape=5)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p5_sd_100 = pm.sample(2000, return_inferencedata=True)

    α_p_post = idata_p5_sd_100.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p5_sd_100.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C3', label=f'model order {5}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    with pm.Model() as model_p3:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=3)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p3 = pm.sample(2000, return_inferencedata=True)

    α_p_post = idata_p3.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p3.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C3', label=f'model order {3}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    waic_p3 = az.waic(idata_p3, scale="deviance")
    print(waic_p3)

    loo_p3 = az.loo(idata_p3, scale="deviance")
    print(loo_p3)
    cmp_df = az.compare({'model_l': idata_p3, 'model_p': idata_p2}, method='BB-pseudo-BMA', ic="waic", scale="deviance")
    print(cmp_df)
    cmp_df = az.compare({'model_l': idata_p3, 'model_p': idata_p5}, method='BB-pseudo-BMA', ic="waic", scale="deviance")
    print(cmp_df)
    cmp_df = az.compare({'model_l': idata_p3, 'model_p': idata_l}, method='BB-pseudo-BMA', ic="waic", scale="deviance")
    print(cmp_df)
    