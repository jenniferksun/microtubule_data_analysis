import numpy as np
import numba
import scipy.stats

import warnings

# Numerical MLE for Gamma distribution
@numba.jit
def log_like_iid_gamma(params, n):
    """Log likelihood for i.i.d. Gamma distributed measurements, parametrized
    by shape, scale."""
    shape, scale = params
    
    if shape <= 0 or scale <= 0:
        return -np.inf
    
    return np.sum(scipy.stats.gamma.logpdf(n, shape, 0, scale))

def mle_iid_gamma(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    multivariate normal measurements, parametrized by mu, Sigma"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_gamma(params, n),
            x0=np.array([0.5, 0.5]),
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
def gen_gamma(shape, scale, size):
    """Draws from the Gamma distribution"""
    
    return np.random.gamma(shape, scale, size=size)


# Numerical MLE for sum of two exponential distributions
@numba.jit
def log_like_iid_two_exp(params, t):
    """Log likelihood for i.i.d. Gamma distributed measurements, parametrized
    by shape, scale."""
    
    b1, delta_b = params
    
    if b1 < 0 or delta_b <= 0:
        return -np.inf
    
    n = len(t)
    
    return n * (np.log(b1) + np.log(delta_b + b1) - np.log(delta_b)) - b1 * np.sum(t) + np.sum(np.log(1 - np.exp(-delta_b * t)))

def mle_iid_two_exp(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    multivariate normal measurements, parametrized by mu, Sigma"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_two_exp(params, n),
            x0=np.array([5, 5]),
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
def gen_two_exp(beta_1, delta_b, size):
    """Draws the total time from the sum of two exponential distributions"""
    beta_2 = delta_b + beta_1
    
    return np.random.exponential(1/beta_1, size) + np.random.exponential(1/beta_2, size)

def cdf_two_exp(t, beta_1, beta_2):
    """Returns the CDF of the proposed model for microtubule catastrophe times"""
    return ((beta_1 * beta_2) / (beta_2 - beta_1)) * ((1 / beta_1) * (1 - np.exp(-beta_1 * t)) - (1 / beta_2) * (1 - np.exp(-beta_2 * t)))


# Other functions
def conf_int(bs_rep):
    """Returns the 95% confidence interval of the given set of drawn bootstrap replicates."""
    return np.percentile(bs_rep, [2.5, 97.5], axis=0)


def draw_parametric_bs_reps_mle(
    mle_fun, gen_fun, data, args=(), size=1, progress_bar=False
):
    """Draw parametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(*params, size)`.
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    params = mle_fun(data, *args)

    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)
    
    res = []
    
    for i in range(size):
        res.append(mle_fun(gen_fun(*params, size=len(data), *args)))
        
    return res