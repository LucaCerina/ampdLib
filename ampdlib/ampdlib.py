# -*- coding: utf-8 -*-
__author__      = "Luca Cerina"
__copyright__   = "Copyright 2023, Luca Cerina"
__credits__     = ["Jeremy Karst", "Steffen Kaiser", "Hans van Gorp"]
__email__       = "lccerina@gmail.com"

"""
This module implements automatic multiscale-based peak detection (AMPD) algorithm as in:
An Efficient Algorithm for Automatic Peak Detection in Noisy Periodic and Quasi-Periodic Signals,
by Felix Scholkmann, Jens Boss and Martin Wolf, Algorithms 2012, 5, 588-603.
"""

from typing import Any, Tuple
import warnings
import multiprocessing

import numpy as np

def get_optimal_size(N:int, maximum_scale:int=None, lsm_limit:float=None, fs:float=None) -> Tuple[int, float, Any]:
	"""Helper function, for many use cases a lsm_limit of 1 may be excessive, particularly for signals with a large size.
	   Calculate scale observable (in samples) given lsm_limit or almost optimal lsm_limit given necessary scale. At least one parameter is required
	   Return also scale in seconds if fs is passed as optional parameter.

		Parameters
		----------
		N (int): Length of the input
		maximum_scale (int, optional): Length in samples to be captured by AMPD. Defaults to None.
		lsm_limit (float, optional): Scale of local maxima search (see ampd function for details). Defaults to None.
		fs (float, optional): Sampling frequency of the signal of interest. Defaults to None.

		Returns
		-------
		int: maximum scale achievable
		float: minimum lsm_limit for maximum scale
		float: time scale. None if fs is None
	"""
	assert (maximum_scale is None) ^ (lsm_limit is None), "Maximum scale or lsm limit is required as a parameter"
	assert (lsm_limit is None) or (0 < lsm_limit <= 1), 'lsm_limit should be comprised between 0 and 1'
	assert (maximum_scale is None) or 0 < maximum_scale <= N, 'maximum_scale should be comprised between 0 and N'
	assert (fs is None) or fs > 0

	if maximum_scale is None:
		_maximum_scale = np.ceil(lsm_limit*N)
		_lsm_limit = lsm_limit
	else:
		_maximum_scale = maximum_scale
		_lsm_limit = maximum_scale/N
	
	if (int(np.ceil(N*_lsm_limit / 2.0)) - 1) <= 2:
		warnings.warn(f"lsm_limit of {_lsm_limit} for scale {_maximum_scale:d} may be too low for signals of size {N:d}. Recommended at least {6/N}.")

	time_scale = _maximum_scale / fs if fs else None

	return _maximum_scale, _lsm_limit, time_scale

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
	N = len(sig_input)
	assert 0 < lsm_limit <= 1, 'lsm_limit should be comprised between 0 and 1'
	assert (int(np.ceil(N*lsm_limit / 2.0)) - 1) > 2, f"lsm_limit is too low for given length {N:d}, recommended lsm_limit > {6/N:f}" 
		
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
			warnings.warn("AMPD: Invalid order, decreasing order")
		while(len(sig_input)%order != 0):
			order -= 1
		if verbose:
			warnings.warn("AMPD: Using order " + str(order))

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
		if hop_length is None:
			hop_length = window_length
		iterations = int((sig_input.shape[0] - window_length) // hop_length) + 1
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

# %% ampd even faster using multiprocessing
def ampd_pool(sig_input:np.ndarray, window_length:int, hop_length:int=None, lsm_limit:float=1, verbose:bool = False, nr_workers:int=None) -> np.ndarray:
    """An even faster version of AMPD which instead of iterating a large signal with windows, the windows are processed in parallel
	
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
            Enable verbosity that will print the number of workers used
		nr_workers: int
			The number of workers to be used. Defaults to None, which will use the maximum number of workers available
        Returns
        -------
        pks: ndarray
            The ordered array of peaks found in sigInput 
    """

    # Assertion checks
    assert 0 < lsm_limit <= 1, 'lsm_limit should be comprised between 0 and 1'
    assert (hop_length is None) or (hop_length < window_length and hop_length > 0), 'hop_length should be smaller than window_length and larger than 0'
    assert (nr_workers is None) or (nr_workers <= multiprocessing.cpu_count() and nr_workers > 0), 'nr_workers should be smaller than the number of available CPUs and larger than 0'

    # Define iterations
    if window_length < sig_input.shape[0]:
        if hop_length is None:
            hop_length = window_length
        iterations = int((sig_input.shape[0] - window_length) // hop_length) + 1
    else:
        window_length = sig_input.shape[0]
        hop_length = window_length
        iterations = 1
        

    # create the arguments for the pool
    args = [(sig_input[(i*hop_length):((i+1)*hop_length + window_length)], lsm_limit) for i in range(iterations)]

	# set the number of workers
    if nr_workers is None:
        nr_workers = multiprocessing.cpu_count()

    if verbose:
        print(f"Using {nr_workers} workers")

	# create a pool of workers using the with statement
    with multiprocessing.Pool(processes=nr_workers) as pool:
		# map the function to the arguments
        pks = pool.starmap(ampd, args)

    # concatenate the results, but first add the offset
    for i in range(iterations):
        pks[i] = pks[i] + i*hop_length

    pks = np.concatenate(pks)

    # Keep only unique values
    pks = np.unique(pks)

    return pks