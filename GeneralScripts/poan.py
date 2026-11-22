import numpy as np
import pandas as pd
#import cv2
import os
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy import stats
from multiprocessing import Pool
def bindata(data0,data1,no_of_bins):
    new_data0 = np.array(data0, dtype=float)
    new_data1 = np.array(data1, dtype=float)
    bin_means, bin_edges, binnumber=stats.binned_statistic(new_data0, new_data1, statistic='std', bins=no_of_bins, range=None)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    return(bin_centers,bin_means)
def meanoftaus(data0,data1,taus2):
    binnedmeans=[]
    for i in range(len(taus2)):
        b0,b1=bindata(data0,data1,taus2[i])
        bmean=np.mean(b1)
        binnedmeans.append(bmean)
        print(i)
    tausarray2=np.array(taus2, dtype=float)
    tausnew2=tausarray2
    
    return(tausnew2,binnedmeans)