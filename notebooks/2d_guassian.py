import numpyro
from numpyro.infer import MCMC, NMC, NUTS
import numpyro.distributions as dist
from jax import random
import seaborn as sns
import jax.numpy as np
import matplotlib.pyplot as plt

def model():
    numpyro.sample('x', dist.MultivariateNormal(loc=np.array([5., 10.]), covariance_matrix=[[3., 0.],
                                                                                            [0., 10.]]))

def run_inifernce(model, rng_key=random.PRNGKey(0)):
    kernel = NMC(model)
    mcmc = MCMC(kernel, 200, 300)
    mcmc.run(rng_key)
    mcmc.print_summary()
    return mcmc.get_samples()

def viz(samples):
    sns.kdeplot(samples['x'][:, 0], samples['x'][:, 1])
    plt.show()


if __name__ == '__main__':
    viz(run_inifernce(model))

