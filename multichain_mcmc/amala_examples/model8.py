from numpy import *
import pymc
from scipy import stats
from scipy.stats import distributions as d

n = 100 

x = random.normal(scale = .2, size = n )
def model_gen():
    
    variables = []

    mu = pymc.Normal("mu", mu = 0, tau = 5**-2 )
    
    variables.append(mu)

    
    obs = pymc.Normal ("obs", mu = mu, tau = .2**-2, observed = True, value = x)

    return variables