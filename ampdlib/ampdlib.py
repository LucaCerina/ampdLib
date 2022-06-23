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

	# Assertion checks
	assert 0 < lsm_limit <= 1, 'lsm_limit should be comprised between 0 and 1' 
		
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
def ampd_fast_sub(sig_input:np.ndarray, order:int = 1, lsm_limit:float = 1, verbose:bool = False) -> np.ndarray:
	"""A slightly faster version of AMPD which divides the signal in 'order' windows

		Parameters
		----------
		sig_input: ndarray
			The 1D signal given as input to the algorithm
		order: int
			The number of windows in which sig_input is divided
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

	# Assertion checks
	assert 0 < lsm_limit <= 1, 'lsm_limit should be comprised between 0 and 1'
	assert order >= 1, 'order should be higher or equal to 1' 

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
	pks = np.zeros((0,), dtype=np.int32)
	for i in range(0, len(sig_input)-N, N):
		if(verbose):
			print(f"\t sector: {i}|{(i+2*N-1)}")
		pks_temp = ampd(sig_input[i:(i+2*N-1)], lsm_limit)
		pks = np.concatenate((pks, pks_temp+i))
		
	# Keep only unique values
	pks = np.unique(pks)
	
	return pks

def ampd_fast(sig_input:np.ndarray, window_length:int, hop_length:int=None, lsm_limit:float=1, verbose:bool = False) -> np.ndarray:
	"""A slightly faster version of AMPD which iterates large signal with windows

		Parameters
		----------
		sig_input: ndarray
			The 1D signal given as input to the algorithm
		window_length: int
			The dimension of the window in samples
		hop_length: int
			The step between windows. Defaults to window_length
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

	# Assertion checks
	assert 0 < lsm_limit <= 1, 'lsm_limit should be comprised between 0 and 1'
	assert (hop_length is None) or (hop_length < window_length and hop_length > 0), 'hop_length should be smaller than window_length and larger than 0'

	# Define iterations
	if window_length < sig_input.shape[0]:
		iterations = int((sig_input.shape[0] - window_length) // hop_length) + 1
		if hop_length is None:
			hop_length = window_length
	else:
		window_length = sig_input.shape[0]
		hop_length = window_length
		iterations = 1

	
	# Loop function calls
	pks = np.zeros((0,), dtype=np.int32)
	for i in range(iterations):
		if(verbose):
			print(f"\t sector: {i*hop_length}|{(i+1)*hop_length + window_length}")
		pks_temp = ampd(sig_input[(i*hop_length):((i+1)*hop_length + window_length)], lsm_limit)
		pks = np.concatenate((pks, pks_temp+i*hop_length))
		
	# Keep only unique values
	pks = np.unique(pks)
	
	return pks