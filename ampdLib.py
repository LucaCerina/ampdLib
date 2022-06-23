# -*- coding: utf-8 -*-
__author__      = "Luca Cerina"
__copyright__   = "Copyright 2022, Luca Cerina"
__credits__     = ["Jeremy Karst", "Steffen Kaiser"]
__email__       = "lccerina@gmail.com"

"""
This module implements automatic multiscale-based peak detection (AMPD) algorithm as in:
An Efficient Algorithm for Automatic Peak Detection in Noisy Periodic and Quasi-Periodic Signals,
by Felix Scholkmann, Jens Boss and Martin Wolf, Algorithms 2012, 5, 588-603.
"""

import numpy as np

# AMPD function
def ampd(sig_input:np.ndarray, lsm_limit:float = 1) -> np.ndarray:
	"""Find the peaks in the signal with the AMPD algorithm.
	
		Original implementation by Felix Scholkmann et al. in
		"An Efficient Algorithm for Automatic Peak Detection in 
		Noisy Periodic and Quasi-Periodic Signals", Algorithms 2012,
		 5, 588-603

		Parameters
		----------
		sig_input: ndarray
			The 1D signal given as input to the algorithm
		lsm_limit: float
			Wavelet transform limit as a ratio of full signal length.
			Valid values: 0-1, the LSM array will no longer be calculated after this point
			  which results in the inability to find peaks at a scale larger than this factor.
			  For example a value of .5 will be unable to find peaks that are of period 
			  1/2 * signal length, a default value of 1 will search all LSM sizes.

		Returns
		-------
		pks: ndarray
			The ordered array of peaks found in sigInput
	"""
		
	# Create preprocessing linear fit	
	sig_time = np.arange(0, len(sig_input))
	
	# Detrend
	dtr_signal = (sig_input - np.polyval(np.polyfit(sig_time, sig_input, 1), sig_time)).astype(float)
	
	N = len(dtr_signal)
	L = int(np.ceil(N*lsm_limit / 2.0)) - 1
	
	# Generate random matrix
	LSM = np.ones([L,N], dtype='uint8')
	
	# Local minima extraction
	for k in range(1, L):
		LSM[k - 1, np.where((dtr_signal[k:N - k - 1] > dtr_signal[0: N - 2 * k - 1]) & (dtr_signal[k:N - k - 1] > dtr_signal[2 * k: N - 1]))[0]+k] = 0
	
	pks = np.where(np.sum(LSM[0:np.argmin(np.sum(LSM, 1)), :], 0)==0)[0]
	return pks


# Fast AMPD		
def ampd_fast_sub(sig_input:np.ndarray, order:int, lsm_limit:float = 1, verbose:bool = False) -> np.ndarray:
	"""A slightly faster version of AMPD which divides the signal in 'order' windows

		Parameters
		----------
		sig_input: ndarray
			The 1D signal given as input to the algorithm
		order: int
			The number of windows in which sigInput is divided
		lsm_limit: float
			Wavelet transform limit as a ratio of full signal length.
			Valid values: 0-1, the LSM array will no longer be calculated after this point
			  which results in the inability to find peaks at a scale larger than this factor.
			  For example a value of .5 will be unable to find peaks that are of period 
			  1/2 * signal length, a default value of 1 will search all LSM sizes.
		verbose: bool
			Enable verbosity while parsing sectors
		Returns
		-------
		pks: ndarray
			The ordered array of peaks found in sigInput 
	"""

	# Check if order is valid (perfectly separable)
	if(len(sig_input)%order != 0):
		if verbose:
			print("AMPD: Invalid order, decreasing order")
		while(len(sig_input)%order != 0):
			order -= 1
		if verbose:
			print("AMPD: Using order " + str(order))

	N = int(len(sig_input) / order / 2)

	# Loop function calls
	for i in range(0, len(sig_input)-N, N):
		if(verbose):
			print("\t sector: " + str(i) + "|" + str((i+2*N-1)))
		pks_temp = ampd(sig_input[i:(i+2*N-1)], lsm_limit)
		if(i == 0):
			pks = pks_temp
		else:
			pks = np.concatenate((pks, pks_temp+i))
		
	# Keep only unique values
	pks = np.unique(pks)
	
	return pks
