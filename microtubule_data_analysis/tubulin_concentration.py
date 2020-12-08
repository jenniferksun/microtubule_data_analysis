import numpy as np
import pandas as pd
import bokeh.io
import iqplot
import os
import pkg_resources
import scipy.stats

from bokeh.plotting import figure
from . import models

import panel as pn
pn.extension()


concentrations = [7, 9, 10, 12, 14]
  
stream = pkg_resources.resource_stream(__name__, 'data/gardner_mt_catastrophe_only_tubulin.csv')
df = pd.read_csv(stream, skiprows=9)

time = pd.DataFrame(columns=["Time to Catastrophe (s)", "Concentration (uM)"]) 

for i in concentrations:
    data = pd.DataFrame(df[str(i) + " uM"].values, columns=["Time to Catastrophe (s)"]).dropna()
    data["Concentration (uM)"] = str(i) + " uM"
    time = time.append(data)
    
concentration_selector = pn.widgets.Select(
    name="Concentration (uM)", options=concentrations, value=12
)


def get_times():
    return time


# Gamma distribution
def get_mle_gamma(df=time):
    mle = []
    
    for i in range(len(concentrations)):
        data = df[df["Concentration (uM)"] == f"{concentrations[i]} uM"]["Time to Catastrophe (s)"].values
        mle.append(models.mle_iid_gamma(data))
    
    mle = pd.DataFrame(mle, columns=["Shape MLE", "Scale MLE"])
    mle["Rate MLE"] = 1 / mle["Scale MLE"]
    
    return pd.concat([pd.DataFrame(concentrations, columns=["Concentration (uM)"]), mle], axis=1)

def get_conf_int_gamma():
    bs_reps = draw_bs_reps_gamma()
    conf_int = []
    
    for i in range(len(concentrations)):
        data = bs_reps[bs_reps["Concentration (uM)"] == concentrations[i]]
        
        temp = np.percentile(data, [2.5, 97.5], axis=0) 
        rate = np.percentile(1 / data["Scale Parameter MLE"], [2.5, 97.5], axis=0) 
        
        conf_int.append([temp[0][0], temp[1][0], temp[0][1], temp[1][1], rate[0], rate[1]])
        
    conf_int = pd.DataFrame(conf_int, columns=["Shape Low", "Shape High", "Scale Low", "Scale High", "Rate Low", "Rate High"])

    return pd.concat([pd.DataFrame(concentrations, columns=["Concentration (uM)"]), conf_int], axis=1)

def get_mle_conf_int_gamma():
    mle = get_mle_gamma()
    conf_int = get_conf_int_gamma()
    conf_int.insert(2, "Shape MLE", mle.iloc[:,1])
    conf_int.insert(5, "Scale MLE", mle.iloc[:,2])
    conf_int.insert(8, "Rate MLE", mle.iloc[:,3])

    return conf_int

def draw_bs_reps_gamma(df=time, path="../data/", rerun=False, size=10000):
    if not rerun:
        stream = pkg_resources.resource_stream(__name__, "data/bs_rep_tubulin_concentrations_gamma.csv")
        return pd.read_csv(stream)

    else:
        data = pd.DataFrame(columns=["Shape Parameter MLE", "Scale Parameter MLE", "Concentration (uM)"])

        for i in range(len(concentrations)):
            times = df[df["Concentration (uM)"] == f"{concentrations[i]} uM"]["Time to Catastrophe (s)"].values

            bs_gamma = models.draw_parametric_bs_reps_mle(
                models.mle_iid_gamma,
                models.gen_gamma,
                times,
                args=(),
                size=size,
                progress_bar=False,
            )

            temp = pd.DataFrame(bs_gamma, columns=["Shape Parameter MLE", "Scale Parameter MLE"])
            temp["Concentration (uM)"] = concentrations[i]
            data = data.append(temp)

        path = os.path.join(path, "bs_rep_tubulin_concentrations_gamma.csv")    
        data.to_csv(path, index = False, header=True)
        return df
    
    
# Sum of two exponential distributions   
def get_mle_two_exp(df=time):
    mle = []
    
    for i in range(len(concentrations)):
        data = df[df["Concentration (uM)"] == f"{concentrations[i]} uM"]["Time to Catastrophe (s)"].values
        beta_1, delta_beta = models.mle_iid_two_exp(data)
        
        mle.append([beta_1, delta_beta + beta_1])
    
    mle = pd.DataFrame(mle, columns=["Beta_1 MLE", "Beta_2 MLE"])
    
    return pd.concat([pd.DataFrame(concentrations, columns=["Concentration (uM)"]), mle], axis=1)
    
def get_conf_int_two_exp():
    bs_reps = draw_bs_reps_two_exp()
    conf_int = []
    
    for i in range(len(concentrations)):
        data = bs_reps[bs_reps["Concentration (uM)"] == concentrations[i]]
        
        temp = np.percentile(data, [2.5, 97.5], axis=0)
        
        conf_int.append([temp[0][0], temp[1][0], temp[0][1], temp[1][1]])
        
    conf_int = pd.DataFrame(conf_int, columns=["Beta_1 Low", "Beta_1 High", "Beta_2 Low", "Beta_2 High"])

    return pd.concat([pd.DataFrame(concentrations, columns=["Concentration (uM)"]), conf_int], axis=1)

def get_mle_conf_int_two_exp():
    mle = get_mle_two_exp()
    conf_int = get_conf_int_two_exp()
    conf_int.insert(2, "Beta_1 MLE", mle.iloc[:,1])
    conf_int.insert(5, "Beta_2 MLE", mle.iloc[:,2])

    return conf_int
    
def draw_bs_reps_two_exp(df=time, path="../data/", rerun=False, size=10000):
    if not rerun:
        stream = pkg_resources.resource_stream(__name__, "data/bs_rep_tubulin_concentrations_two_exp.csv")
        return pd.read_csv(stream)

    else:
        data = pd.DataFrame(columns=["Beta_1 Parameter MLE", "Delta_Beta Parameter MLE", "Concentration (uM)"])

        for i in range(len(concentrations)):
            times = df[df["Concentration (uM)"] == f"{concentrations[i]} uM"]["Time to Catastrophe (s)"].values

            bs_two_exp = models.draw_parametric_bs_reps_mle(
                models.mle_iid_two_exp,
                models.gen_two_exp,
                times,
                args=(),
                size=size,
                progress_bar=False,
            )

            temp = pd.DataFrame(bs_two_exp, columns=["Beta_1 Parameter MLE", "Delta_Beta Parameter MLE"])
            temp["Concentration (uM)"] = concentrations[i]
            data = data.append(temp)


        data.columns = ["Beta_1 Parameter MLE", "Beta_2 Parameter MLE", "Concentration (uM)"]
        data["Beta_2 Parameter MLE"] = data["Beta_2 Parameter MLE"] + data["Beta_1 Parameter MLE"]

        path = os.path.join(path, "bs_rep_tubulin_concentrations_two_exp.csv")    
        data.to_csv(path, index = False, header=True)
        return df
        
        
# Plots 
palette = bokeh.palettes.Bokeh[7]
colors = [palette[0], palette[2], palette[4], palette[5], palette[6]]

def plot_box():
    p = iqplot.box(
        data=time,
        q="Time to Catastrophe (s)",
        cats="Concentration (uM)",
        title=f"Box plot of Varying Tubulin Concentrations Catastrophe Times",
        palette=colors
    )
    
    bokeh.io.show(p)
    
def plot_stripbox():
    p = iqplot.stripbox(
        data=time,
        q="Time to Catastrophe (s)",
        cats="Concentration (uM)",
        title="Stripbox plot of Varying Tubulin Concentrations Catastrophe Times",
        palette=colors,
        jitter=True
    )
    
    bokeh.io.show(p)

def plot_ecdf_all():
    p = iqplot.ecdf(
        data=time,
        q="Time to Catastrophe (s)",
        cats="Concentration (uM)",
        title="ECDF of Varying Tubulin Concentrations Catastrophe Times",
        palette=colors
    )
    
    bokeh.io.show(p)
    
def plot_mle_gamma():
    mle = get_mle_gamma()
    
    p = figure(
        width=450,
        height=350,
        x_axis_label="Time to Catastrophe (s)", 
        y_axis_label="Cumulative distribution",
        title="MLE Gamma Distribution for Varying Tubulin Concentrations"
    )
    
    x = np.linspace(0, 2000)

    for i in range(len(concentrations)):
        data = mle[mle["Concentration (uM)"] == concentrations[i]]
        
        cdf = scipy.stats.gamma.cdf(x, data["Shape MLE"], 0, data["Scale MLE"])
        p.line(x, cdf, color=colors[i], legend_label=f"{concentrations[i]} uM")

    bokeh.io.show(p)
    
def plot_mle_conf_int_gamma(concentration=12):
    df = get_mle_conf_int_gamma()
    data = df[df["Concentration (uM)"] == concentration]

    p = iqplot.ecdf(
        data=time[time["Concentration (uM)"] == f"{concentration} uM"],
        q="Time to Catastrophe (s)",
        cats="Concentration (uM)",
        title=f"ECDF and Gamma Distribution of {concentration} uM Tubulin Catastrophe Times",
        conf_int=True,
        palette=[bokeh.colors.named.lightgray.to_hex()],
        plot_width=600,
        plot_height=400
    )

    # Plot CDFs of time to catastrophe.
    x = np.linspace(0, 2000)

    cdf = scipy.stats.gamma.cdf(x, data["Shape MLE"], 0, data["Scale MLE"])
    cdf_lo = scipy.stats.gamma.cdf(x, data["Shape Low"], 0, data["Scale Low"])
    cdf_hi = scipy.stats.gamma.cdf(x, data["Shape High"], 0, data["Scale High"])

    p.line(x, cdf, color=colors[3], legend_label="MLE of Gamma Distribution Parameters")
    p.line(x, cdf_lo, color=colors[2], legend_label="Upper Bound Gamma (Lower Bound MLE)")
    p.line(x, cdf_hi, color=colors[4], legend_label="Lower Bound Gamma (Upper Bound MLE)")
    
    return p

def plot_mle_conf_int_two_exp(concentration=12):
    df = get_mle_conf_int_two_exp()
    data = df[df["Concentration (uM)"] == concentration]

    p = iqplot.ecdf(
        data=time[time["Concentration (uM)"] == f"{concentration} uM"],
        q="Time to Catastrophe (s)",
        cats="Concentration (uM)",
        title=f"ECDF and two Exponential Distributions of {concentration} uM Tubulin Catastrophe Times",
        conf_int=True,
        palette=[bokeh.colors.named.lightgray.to_hex()],
        plot_width=600,
        plot_height=400
    )

    # Plot CDFs of time to catastrophe.
    t = np.linspace(0, 2000)
    f_mle = models.cdf_two_exp(t, data.iloc[0]["Beta_1 MLE"], data.iloc[0]["Beta_2 MLE"])
    f_lo = models.cdf_two_exp(t, data.iloc[0]["Beta_1 Low"], data.iloc[0]["Beta_2 Low"])
    f_hi = models.cdf_two_exp(t, data.iloc[0]["Beta_1 High"], data.iloc[0]["Beta_2 High"])

    p.line(t, f_mle, color=colors[3], legend_label="MLE of Distribution Parameters")
    p.line(t, f_lo, color=colors[4], legend_label="Lower Bound Distribution (Lower Bound MLE)")
    p.line(t, f_hi, color=colors[2], legend_label="Upper Bound Distribution (Upper Bound MLE)")
    
    return p


# Panel for Gamma distributions
@pn.depends(
    concentration_selector.param.value
)    
def make_ecdf_plot_interactive_gamma(concentration):
    return plot_mle_conf_int_gamma(concentration)

def plot_concentration_panel_gamma():
    widgets = pn.Column(
        concentration_selector,
        width=300
    )

    ecdf = pn.Row(make_ecdf_plot_interactive_gamma, pn.Spacer(width=20), widgets)

    return pn.Column(ecdf)


# Panel for sum of two exponential distributions
@pn.depends(
    concentration_selector.param.value
)    
def make_ecdf_plot_interactive_two_exp(concentration):
    return plot_mle_conf_int_two_exp(concentration)


def plot_concentration_panel_two_exp():
    widgets = pn.Column(
        concentration_selector,
        width=300
    )

    ecdf = pn.Row(make_ecdf_plot_interactive_two_exp, pn.Spacer(width=20), widgets)

    return pn.Column(ecdf)