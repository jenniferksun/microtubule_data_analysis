"""Functions used in homework 6.1 to generate ECDFs with 95% confidence intervals,
compute bounds for confidence intervals using the Dvoretzky-Kiefer-Wolfowitz Inequality,
compute confidence intervals for the means of tubulin data (according to the Central
Limit Theorem and for plugin estimates), and ecdf value generation."""

import os, sys
import numpy as np
import pandas as pd
import bokeh.io
import bokeh.plotting
import iqplot
import numba
import scipy.stats
import math
from numba import jit
import pkg_resources
from .gen_ecdf import *


# Load in tidy tubulin data.
df = load_data()

# Separate labeled and unlabeled tubulin times to catastrophe (s) as numpy arrays.
labeled = df['time to catastrophe (s)'][df['labeled'] == True].to_numpy()
unlabeled = df['time to catastrophe (s)'][df['labeled'] == False].to_numpy()

# Create colors for plots.
palette = bokeh.palettes.Bokeh[7]
colors = [palette[0], palette[2], palette[4], palette[5], palette[6]]


def plot_ecdf_with_conf_ints_labeled_v_unlabeled():
    """Plots ECDFs of times to catastrophe (s) for labeled and unlabled tubulin
    with 95% confidence intervals."""
    
    df2 = pd.DataFrame(labeled, columns=["Time to Catastrophe (s)"])
    df2["labeled"] = "Labeled"
    df3 = pd.DataFrame(unlabeled, columns=["Time to Catastrophe (s)"])
    df3["labeled"] = "Unlabeled"

    cat_df = df2.append(df3)
    
    p = iqplot.ecdf(
            data = cat_df,
            q = "Time to Catastrophe (s)",
            cats = ["labeled"], 
            conf_int = True,
            palette=[palette[0], palette[5]]
            )
    bokeh.io.show(p)


def _draw_bs_sample(data):
    """Draws a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))


def _draw_bs_reps_mean(data, size=1):
    """Draws boostrap replicates of the mean from a 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(_draw_bs_sample(data))
    return out


def print_plugin_mean_conf_ints_labeled_v_unlabeled():
    """Prints the 95% confidence intervals for the plug-in 
    estimates of the means of the labeled and unlabeled tubulin
    data. Confidence intervals are computed using 10000 bootstrap
    replicates."""
    
    labeled_mean = _draw_bs_reps_mean(labeled, size=10000)
    unlabeled_mean = _draw_bs_reps_mean(unlabeled, size=10000)

    # Compute the 95% confidence intervals.
    mean_labeled_conf_int = np.percentile(labeled_mean, [2.5, 97.5])
    mean_unlabeled_conf_int = np.percentile(unlabeled_mean, [2.5, 97.5])

    print("""
    Mean time to catastrophe for labeled tubulin 95% confidence interval (s):   [{0:.2f}, {1:.2f}]
    Mean time to catastrophe for unlabeled tubulin 95% confidence interval (s): [{2:.2f}, {3:.2f}]
    """.format(*(tuple(mean_labeled_conf_int) + tuple(mean_unlabeled_conf_int))))


def ks_2samp_labeled_v_unlabeled():
    """Conducts a 2-sample KS test with labeled and unlabeled tubulin data."""
    return scipy.stats.ks_2samp(labeled, unlabeled)


def print_mean_per_CLT_conf_ints_labeled_v_unlabeled():
    """Prints the 95% confidence intervals for the means of the 
    labeled and unlabeled tubulin data, assuming the means are normally 
    distributed (by the Central Limit Theorem (CLT))."""
    
    # Calculating confidence interval for labeled tubulin
    labeled_mu = np.mean(labeled)
    labeled_sigma = np.std(labeled)
    labeled_conf_int = scipy.stats.norm.interval(0.95, loc=labeled_mu, scale=labeled_sigma/np.sqrt(len(labeled)))

    # Calculating confidence interval for unlabeled tubulin
    unlabeled_mu = np.mean(unlabeled)
    unlabeled_sigma = np.std(unlabeled)
    unlabeled_conf_int = scipy.stats.norm.interval(0.95, loc=unlabeled_mu, scale=unlabeled_sigma/np.sqrt(len(unlabeled)))

    print("""
    Mean time to catastrophe for labeled tubulin 95% confidence interval (s):   [{0:.2f}, {1:.2f}]
    Mean time to catastrophe for unlabeled tubulin 95% confidence interval (s): [{2:.2f}, {3:.2f}]
    """.format(*((labeled_conf_int) + (unlabeled_conf_int))))

def _ecdf(x, data):
    """Returns the y values for plotting the ECDF for a given one-dimensional set of data and arbitrary x values."""
    y = []
    for i in x:
        y.append(sum(k <= i for k in data)/len(data))
    return y


def plot_ecdf_with_conf_ints_and_DKW_bounds_labeled_v_unlabeled():
    """Plots ECDFs of times to catastrophe (s) for labeled and unlabled tubulin
    with 95% confidence intervals and upper and lower bounds for the confidence
    intervals, computed using the Dvoretzky-Kiefer-Wolfowitz Inequality."""
    
    # Define alpha with respect to the 95% confidence interval.
    alpha = 0.05
    
    # Compute epsilon for the DKW bounds for labeled and unlabeled tubulin.
    eps_labeled = np.sqrt(1/(2 * len(labeled)) * np.log(2/alpha))
    eps_unlabeled = np.sqrt(1/(2 * len(unlabeled)) * np.log(2/alpha))

    # Create ranges of times to catastrophe for labeled and unlabeled data.
    x_labeled = list(range(math.floor(min(labeled)), math.ceil(max(labeled))))
    x_unlabeled = list(range(math.floor(min(unlabeled)), math.ceil(max(unlabeled))))

    # Find ECDF values for labeled and unlabeled.
    F_x_labeled = np.array(_ecdf(x_labeled, labeled))
    F_x_unlabeled = np.array(_ecdf(x_unlabeled, unlabeled))

    # Find DKW lower bound for confidence interval of labeled.
    lower_bounds_labeled = F_x_labeled - eps_labeled
    lower_bounds_labeled[lower_bounds_labeled < 0] = 0

    # Find DKW upper bound for confidence interval of labeled.
    upper_bounds_labeled = F_x_labeled + eps_labeled
    upper_bounds_labeled[upper_bounds_labeled > 1] = 1

    # Find DKW lower bound for confidence interval of unlabeled.
    lower_bounds_unlabeled = F_x_unlabeled - eps_unlabeled
    lower_bounds_unlabeled[lower_bounds_unlabeled < 0] = 0

    # Find DKW upper bound for confidence interval of unlabeled.
    upper_bounds_unlabeled = F_x_unlabeled + eps_unlabeled
    upper_bounds_unlabeled[upper_bounds_unlabeled > 1] = 1
    
    # Re-populate labeled column for plot legend.
    df_temp = df.copy()
    df_temp = df_temp.replace(True, "Labeled Tubulin")
    df_temp = df_temp.replace(False, "Unlabeled Tubulin")
    

    # Plot ECDFs of times to catastrophe for labeled and unlabeled.
    p = iqplot.ecdf(
        data = df_temp,
        q = "time to catastrophe (s)",
        cats = ["labeled"], 
        conf_int = True,
        palette=[palette[0], palette[5]],
        title = "Lower Bound vs Upper bound of Labeled and Unlabeled tubulin"
        )
    
    # Plot labeled bounds
    p.line(x_labeled, lower_bounds_labeled, color=palette[1], legend_label="Labeled lower bound", line_width=1.5)
    p.line(x_labeled, upper_bounds_labeled, color=palette[2], legend_label="Labeled upper bound", line_width=1.5)

    # Plot unlabeled bounds
    p.line(x_unlabeled, lower_bounds_unlabeled, color=palette[4], legend_label="Unlabeled lower bound", line_width=1.5)
    p.line(x_unlabeled, upper_bounds_unlabeled, color=palette[6], legend_label="Unlabeled upper bound", line_width=1.5)

    p.legend.location = "bottom_right"
    p.legend.click_policy="hide"
    bokeh.io.show(p)