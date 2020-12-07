"""Functions used in homework 5.2 used to simulate microtubule catastrophe
events using Exponential and Gamma distribution"""
import numpy as np
import bokeh.io
import bokeh.plotting
import iqplot
import matplotlib.pyplot as plt
from bokeh.plotting import figure
import bokeh.palettes
import pandas as pd
from . import gen_ecdf

from scipy.stats import norm

def simulate_experiment_with_exponential():
    """Generate plot of ECDFs of catastrophe processes modeled as two successive
    Poisson events, for different ratios of rates of the Poisson processes"""
    # Creating colors
    palette = bokeh.palettes.Bokeh[7]
    colors = [palette[0], palette[2], palette[4], palette[5], palette[6]]

    n = 150

    # Define the rate arrivals for pairs of Poisson processes.
    # The rate arrivals pairs respectively have distinct ratios.
    B_1 = [1, 1, 1, 1, 1]
    B_2 = [0.125, 0.25, 0.5, 1, 2]

    # Initialize a dataframe.
    df = pd.DataFrame()

    # For each pair of rate arrivals, we compute the time to catastrophe over n events.
    for i in range(len(B_1)):
        # Sample time for the first process to occur.
        b_1 = np.random.exponential(1/B_1[i], n)

        # Sample time for the second process to occur.
        b_2 = np.random.exponential(1/B_2[i], n)

        temp = pd.DataFrame()

        # Compute and store the total time to catastrophe.
        temp["(Time to Catastrophe) * B_1"] = (b_1 + b_2) * (1 / B_1[i])

        # Track the corresponding ratio of B_2 to B_1.
        temp["ratio"] = str(B_2[i] / B_1[i])
        df = df.append(temp)

    # Plot an ECDF of the times to catastrophe for each pair of rate arrivals.
    p = iqplot.ecdf(
        data=df,
        q="(Time to Catastrophe) * B_1",
        cats=["ratio"],
        palette=colors
        )
    p.legend.title = "beta2/beta1 ratio"
    p.legend.click_policy="hide"
    return p


def compare_ecdf_with_analytical_cdf():
    """Generate overlayed plots of the randomly generated ECDF and the
    analytical CDF of the catastrophe with rates of 1 and 0.25 for the first
    and second processes respectively"""
    # Define the rate arrivals of the two Poisson processes.
    B_1 = 1
    B_2 = 0.25

    # Define the number of events.
    n = 150

    # Sample time for first process to occur.
    b_1 = np.random.exponential(1/B_1, n)

    # Sample time for second process to occur.
    b_2 = np.random.exponential(1/B_2, n)

    # Compute total time to catastrophe.
    time_to_catastrophe = (b_1 + b_2) * (1 / B_1)

    # Compute ECDF of times to catastrophe.
    (x, y) = gen_ecdf.ecdf_vals(time_to_catastrophe)


    # Compute analytical CDF for times to catastrophe.
    t = np.linspace(0, 50, n)
    F = ((B_1 * B_2) / (B_2 - B_1)) * ((1 / B_1) * (1 - np.exp(-B_1 * t)) -
        (1 / B_2) * (1 - np.exp(-B_2 * t)))


    # Plot a CDF and ECDF of the times to catastrophe for B_2 / B_1 = 0.25.
    p = figure(plot_width=400, plot_height=400, x_axis_label="(Time to " +
        "Catastrophe) * B_1", y_axis_label="Cumulative Distribution",
        title="ECDF vs Analytical CDF for (B_2 / B_1) = 0.25")

    p.line(t, F, color="blue", legend_label="Analytical CDF for B_2 / B_1 = 0.25")
    p.line(x, y, color="red", legend_label="ECDF for B_2 / B_1 = 0.25")
    p.legend.location = "bottom_right"
    return p
