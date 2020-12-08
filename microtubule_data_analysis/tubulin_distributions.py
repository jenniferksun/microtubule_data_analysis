import numpy as np
import pandas as pd

import bokeh.io
import iqplot
import os
import pkg_resources
import scipy.stats

from . import models


stream = pkg_resources.resource_stream(__name__, 'data/gardner_time_to_catastrophe_dic_tidy.csv')
df = pd.read_csv(stream)

labeled = df[df["labeled"] == True]

def get_labeled():
    return labeled

# Gamma distribution
def get_mle_gamma(df=labeled):
    data = df["time to catastrophe (s)"].values
    mle = models.mle_iid_gamma(data)
    
    mle = pd.DataFrame([mle], columns=["Shape MLE", "Scale MLE"])
    mle["Rate MLE"] = 1 / mle["Scale MLE"]
    
    return mle

def get_conf_int_gamma():
    data = draw_bs_reps_gamma()
    conf_int = []
    
    temp = np.percentile(data, [2.5, 97.5], axis=0) 
    rate = np.percentile(1 / data["Scale Parameter MLE"], [2.5, 97.5], axis=0) 

    conf_int = [temp[0][0], temp[1][0], temp[0][1], temp[1][1], rate[0], rate[1]]
        
    return pd.DataFrame([conf_int], columns=["Shape Low", "Shape High", "Scale Low", "Scale High", "Rate Low", "Rate High"])

def get_mle_conf_int_gamma():
    mle = get_mle_gamma()
    conf_int = get_conf_int_gamma()
    conf_int.insert(1, "Shape MLE", mle.iloc[:,0])
    conf_int.insert(4, "Scale MLE", mle.iloc[:,1])
    conf_int.insert(7, "Rate MLE", mle.iloc[:,2])

    return conf_int

def draw_bs_reps_gamma(df=labeled, path="../data/", rerun=False, size=10000):
    if not rerun:
        stream = pkg_resources.resource_stream(__name__, "data/bs_rep_tubulin_labeled_gamma.csv")
        return pd.read_csv(stream)

    else:

        bs_gamma = models.draw_parametric_bs_reps_mle(
            models.mle_iid_gamma,
            models.gen_gamma,
            df["time to catastrophe (s)"].values,
            args=(),
            size=size,
            progress_bar=False,
        )

        data = pd.DataFrame(bs_gamma, columns=["Shape Parameter MLE", "Scale Parameter MLE"])

        path = os.path.join(path, "bs_rep_tubulin_labeled_gamma.csv")    
        data.to_csv(path, index = False, header=True)
        return data
    
# Sum of two exponential distributions   
def get_mle_two_exp(df=labeled):
    data = df["time to catastrophe (s)"].values
    beta_1, delta_beta = models.mle_iid_two_exp(data)

    mle = [beta_1, delta_beta + beta_1]

    return pd.DataFrame([mle], columns=["Beta_1 MLE", "Beta_2 MLE"])

def get_conf_int_two_exp():
    data = draw_bs_reps_two_exp()
    conf_int = []
    
    temp = np.percentile(data, [2.5, 97.5], axis=0) 

    conf_int = [temp[0][0], temp[1][0], temp[0][1], temp[1][1]]

    return pd.DataFrame([conf_int], columns=["Beta_1 Low", "Beta_1 High", "Beta_2 Low", "Beta_2 High"])
    
def get_mle_conf_int_two_exp():
    mle = get_mle_two_exp()
    conf_int = get_conf_int_two_exp()
    conf_int.insert(1, "Beta_1 MLE", mle.iloc[:,0])
    conf_int.insert(4, "Beta_2 MLE", mle.iloc[:,1])

    return conf_int
    
    
def draw_bs_reps_two_exp(df=labeled, path="../data/", rerun=False, size=10000):
    if not rerun:
        stream = pkg_resources.resource_stream(__name__, "data/bs_rep_tubulin_labeled_two_exp.csv")
        return pd.read_csv(stream)

    else:
        bs_two_exp = models.draw_parametric_bs_reps_mle(
            models.mle_iid_two_exp,
            models.gen_two_exp,
            df["time to catastrophe (s)"].values,
            args=(),
            size=size,
            progress_bar=False,
        )

        data = pd.DataFrame(bs_two_exp, columns=["Beta_1 Parameter MLE", "Delta_Beta Parameter MLE"])

        data.columns = ["Beta_1 Parameter MLE", "Beta_2 Parameter MLE"]
        data["Beta_2 Parameter MLE"] = data["Beta_2 Parameter MLE"] + data["Beta_1 Parameter MLE"]

        path = os.path.join(path, "bs_rep_tubulin_labeled_two_exp.csv")    
        data.to_csv(path, index = False, header=True)
        return data
    
    
# Plots    
palette = bokeh.palettes.Bokeh[7]
colors = [palette[0], palette[2], palette[4], palette[5], palette[6]]
  
def plot_mle_conf_int_gamma():
    df = get_mle_conf_int_gamma()
    
    p = iqplot.ecdf(
        data=labeled,
        q ="time to catastrophe (s)",
        cats = "labeled",
        title=f"ECDF and Gamma Distribution of Labeled Tubulin Catastrophe Times",
        conf_int=True,
        palette=[bokeh.colors.named.lightgray.to_hex()],
        plot_width=600,
        plot_height=400
    )

    # Plot CDFs of time to catastrophe.
    x = np.linspace(0, 2000)

    cdf = scipy.stats.gamma.cdf(x, df["Shape MLE"], 0, df["Scale MLE"])
    cdf_lo = scipy.stats.gamma.cdf(x, df["Shape Low"], 0, df["Scale Low"])
    cdf_hi = scipy.stats.gamma.cdf(x, df["Shape High"], 0, df["Scale High"])

    p.line(x, cdf, color=colors[3], legend_label="MLE of Gamma Distribution Parameters")
    p.line(x, cdf_lo, color=colors[2], legend_label="Upper Bound Gamma (Lower Bound MLE)")
    p.line(x, cdf_hi, color=colors[4], legend_label="Lower Bound Gamma (Upper Bound MLE)")
    
    bokeh.io.show(p)

def plot_mle_conf_int_two_exp(concentration=12):
    df = get_mle_conf_int_two_exp()

    p = iqplot.ecdf(
        data=labeled,
        q ="time to catastrophe (s)",
        cats = "labeled",
        title=f"ECDF and Sum of Two Exponential Distributions of Labeled Tubulin Catastrophe Times",
        conf_int=True,
        palette=[bokeh.colors.named.lightgray.to_hex()],
        plot_width=600,
        plot_height=400
    )

    # Plot CDFs of time to catastrophe.
    t = np.linspace(0, 2000)
    f_mle = models.cdf_two_exp(t, df.iloc[0]["Beta_1 MLE"], df.iloc[0]["Beta_2 MLE"])
    f_lo = models.cdf_two_exp(t, df.iloc[0]["Beta_1 Low"], df.iloc[0]["Beta_2 Low"])
    f_hi = models.cdf_two_exp(t, df.iloc[0]["Beta_1 High"], df.iloc[0]["Beta_2 High"])

    p.line(t, f_mle, color=colors[3], legend_label="MLE of Distribution Parameters")
    p.line(t, f_lo, color=colors[4], legend_label="Lower Bound Distribution (Lower Bound MLE)")
    p.line(t, f_hi, color=colors[2], legend_label="Upper Bound Distribution (Upper Bound MLE)")
    
    bokeh.io.show(p)
