# -*- coding: utf-8 -*-
__author__      = "Luca Cerina"
__copyright__   = "Copyright 2022, Luca Cerina"
__email__       = "lccerina@gmail.com"

# import sys
# from time import perf_counter
# import numpy as np
# from scipy.io import loadmat
# sys.path.append('../')
# import ampdLib

# if __name__ == "__main__":
#     print("Loading data")
#     real_data = loadmat('data.mat')

#     print("Find peaks")
#     N = 30000
#     input_data = real_data['ecg_signal'][0:N,0].flatten()
#     tStart = perf_counter()
#     ampd_peaks = ampdLib.ampd_fast(input_data)
#     tEnd = perf_counter()
#     print("Samples: {}".format(N))
#     print("Peaks found in: {} seconds".format(tEnd-tStart))

#     print("Test results")
#     # Find misplaced peaks
#     error_peaks = np.sum((ampd_peaks - real_data['real_peaks'][0][0:ampd_peaks.shape[0]]) != 0)
#     if(error_peaks == 0):
#         print("Test passed")
#     else:
#         print("Total error {}".format(error_peaks))

import unittest

import ampdlib
import numpy as np
from scipy.io import loadmat


class TestLibrary(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data = loadmat('data.mat')
        N = 30000
        self.input_data = self.test_data['ecg_signal'][0:N,0].flatten()
        self.real_peaks = self.test_data['real_peaks'][0]
        return super().setUp()

    def test_detection(self):
        ampd_peaks = ampdlib.ampd_fast(self.input_data, window_length=2000, hop_length=1000)
        error_peaks = np.sum((ampd_peaks - self.real_peaks[0:ampd_peaks.shape[0]]) != 0)
        self.assertEqual(error_peaks, 0)

    def test_no_hop_length(self):
        ampd_peaks = ampdlib.ampd_fast(self.input_data, window_length=2000)
        error_peaks = np.sum((ampd_peaks - self.real_peaks[0:ampd_peaks.shape[0]]) != 0)
        self.assertEqual(error_peaks, 0)

    def test_assertions(self):
        with self.assertRaises(AssertionError):
            ampdlib.ampd(self.input_data, lsm_limit=-1)
        with self.assertRaises(AssertionError):
            ampdlib.ampd(self.input_data, lsm_limit=2)
        with self.assertRaises(AssertionError):
            ampdlib.ampd_fast(self.input_data, 2000, -1)
        with self.assertRaises(AssertionError):
            ampdlib.ampd_fast(self.input_data, 2000, 2100)

if __name__ == "__main__":
    unittest.main()