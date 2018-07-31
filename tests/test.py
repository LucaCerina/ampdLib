# -*- coding: utf-8 -*-
__author__      = "Luca Cerina"
__copyright__   = "Copyright 2018, Luca Cerina"
__email__       = "luca.cerina@polimi.it"

import sys
from time import perf_counter
import numpy as np
from scipy.io import loadmat
sys.path.append('../')
import ampdLib

if __name__ == "__main__":
    print("Loading data")
    real_data = loadmat('data.mat')

    print("Find peaks")
    N = 30000
    input_data = real_data['ecg_signal'][0:N,0].flatten()
    tStart = perf_counter()
    ampd_peaks = ampdLib.ampd_v2(input_data)
    tEnd = perf_counter()
    print("Samples: {}".format(N))
    print("Peaks found in: {} seconds".format(tEnd-tStart))

    print("Test results")
    # Find misplaced peaks
    error_peaks = np.sum((ampd_peaks - real_data['real_peaks'][0][0:ampd_peaks.shape[0]]) != 0)
    if(error_peaks == 0):
        print("Test passed")
    else:
        print("Total error {}".format(error_peaks))
