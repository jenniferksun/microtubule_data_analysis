import numpy as np
import pandas as pd
import os
import pkg_resources

from . import models

concentrations = [7, 9, 10, 12, 14]
  
stream = pkg_resources.resource_stream(__name__, 'data/gardner_mt_catastrophe_only_tubulin.csv')
df = pd.read_csv(stream, skiprows=9)

time = pd.DataFrame(columns=["Time to Catastrophe (s)", "Concentration (uM)"]) 

for i in concentrations:
    data = pd.DataFrame(df[str(i) + " uM"].values, columns=["Time to Catastrophe (s)"]).dropna()
    data["Concentration (uM)"] = str(i) + " uM"
    time = time.append(data)
    
def get_times():
    return time

def get_mle_gamma(df=time):
    mle = []
    
    for i in range(len(concentrations)):
        data = df[df["Concentration (uM)"] == f"{concentrations[i]} uM"]["Time to Catastrophe (s)"].values
        mle.append(models.mle_iid_gamma(data))
    
    mle = pd.DataFrame(mle, columns=["Shape Parameter MLE", "Scale Parameter MLE"])
    mle["Rate Parameter MLE"] = 1 / mle["Scale Parameter MLE"]
    mle["Concentration (uM)"] = sorted(concentrations)
    
    return mle

# def get_conf_int(concentration):
#     path = f"../data/bs_rep_tubulin_{concentration}_gamma.csv"
        
#     return models.conf_int(pd.read_csv(path))

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
        