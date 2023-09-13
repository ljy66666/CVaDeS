import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import torch

def km_logrank_picture(pred,time,even,name):


    median=torch.median(pred)

    pred_below_median = pred[pred <= median]
    pred_above_median = pred[pred > median]

    time_below_median = time[pred <= median]
    time_above_median = time[pred > median]

    even_below_median = even[pred <= median]
    even_above_median = even[pred > median]

    kmf_group1 = KaplanMeierFitter()
    kmf_group1.fit(time_below_median, even_below_median)

    kmf_group2 = KaplanMeierFitter()
    kmf_group2.fit(time_above_median, even_above_median)

    # log-rank
    results = logrank_test(time_below_median, time_above_median, even_below_median, even_above_median)

    # get p-value
    p_value = results.p_value
    
    len_group1=len(time_below_median)
    len_group2=len(time_above_median)
    


    plt.figure(figsize=(8, 6))
    kmf_group1.plot(color='blue', label=f'low risk ,N={len_group1}')
    kmf_group2.plot(color='red', label=f'high risk ,N={len_group2}')
    plt.xlabel('Time')

    plt.ylabel('Survival Probability')
    if p_value < 0.0001:
        plt.title('Kaplan-Meier Survival Curve , p-value < 0.0001')

    plt.title('Kaplan-Meier Survival Curve  p-value: {:.4f}'.format(p_value))
    plt.legend()

    
    plt.savefig(f'{name}_logrank-curve.png')
