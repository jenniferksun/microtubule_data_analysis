"""Functions used in homework 2.2 to generate ECDFs using iqplot and personal
implementation of ecdf value generation"""

import os, sys
import numpy as np
import pandas as pd
import bokeh.io
import bokeh.plotting
import pkg_resources


def load_data():
    data_path = "data/"
    data_path = pkg_resources.resource_stream(__name__, 'data/gardner_time_to_catastrophe_dic_tidy.csv')
    df = pd.read_csv(data_path)
    return df


def ecdf_vals(data):
    """Returns the x and y values for plotting the ECDF for a
    given one-dimensional set of data."""

    # Sort the values in the data
    x = np.sort(data)
    n = x.size

    # Compute the fraction of data points <= x
    y = np.arange(1, n + 1) / n
    return (x, y)


def plot_ecdf_labeled_v_unlabeled():
    df = load_data()
    # Slice out the time to catastrophe for labeled and unlabeled tubulin
    labeled = df['time to catastrophe (s)'][df['labeled'] == True].to_numpy()
    unlabeled = df['time to catastrophe (s)'][df['labeled'] == False].to_numpy()

    # Compute the x and y for plotting the ECDF of labeled an unlabed tubulin
    (x_labeled, y_labeled) = ecdf_vals(labeled)
    (x_unlabeled, y_unlabeled) = ecdf_vals(unlabeled)

    # Plot the ECDF
    p = bokeh.plotting.figure(
        width=400,
        height=300,
        x_axis_label="Catastophe time (s)",
        y_axis_label="Cumulative distribution",
        title="Labeled and unlabeled tubulin catastrophe time ECDFs"
    )

    # Plotting glyphs for the ECDF of labeled tubulin
    p.circle(x_labeled, y_labeled, color="orange", legend_label="Labeled")

    # Plotting glyphs for the ECDF of unlabeled tubulin
    p.circle(x_unlabeled, y_unlabeled, legend_label="Unlabeled")

    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    return p
