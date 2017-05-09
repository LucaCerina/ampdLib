# -*- coding: utf-8 -*-
__author__      = "Luca Cerina"
__copyright__   = "Copyright 2017, Luca Cerina"
__email__       = "luca.cerina@polimi.it"


import numpy as np

# AMPD function
def ampd(sigInput):
	"""Find the peaks in the signal with the AMPD algorithm

		Parameters
		----------
		sigInput: ndarray
			The 1D signal given as input to the algorithm

		Returns
		-------
		pks: ndarray
			The ordered array of peaks found in sigInput
	"""

	sigInput = sigInput.reshape(len(sigInput), 1)
		
	# Create preprocessing linear fit	
	sigTime = np.arange(0, len(sigInput))
	
	fitPoly = np.polyfit(sigTime, sigInput, 1)
	sigFit = np.polyval(fitPoly, sigTime)
	
	# Detrend
	dtrSignal = sigInput - sigFit
	
	N = len(dtrSignal)
	L = int(np.ceil(N / 2.0)) - 1
	
	# Generate random matrix
	LSM = np.random.uniform(1.0, 2.0, size = (L,N)) # uniform + alpha = 1
	
	# Local minima extraction
	for k in np.arange(1, L):
		for i in range(k+1, N-k):
			if((sigInput[i-1, 0] > sigInput[i-k-1, 0]) 
			& (sigInput[i-1, 0] > sigInput[i+k-1, 0])):
				LSM[k-1, i-1] = 0
	
	# Find minima				
	G = np.sum(LSM, 1)
	l = np.where(G == G.min())[0]
	
	LSM = LSM[0:l, :]
	
	S = np.std(LSM, 0)

	pks = np.flatnonzero(S == 0)
	return pks

# Fast AMPD		
def ampdFast(sigInput, order):
	"""A slightly faster version of AMPD which divides the signal in 'order' windows

		Parameters
		----------
		sigInput: ndarray
			The 1D signal given as input to the algorithm
		order: int
			The number of windows in which sigInput is divided

		Returns
		-------
		pks: ndarray
			The ordered array of peaks found in sigInput 
	"""

	# Check if order is valid (perfectly separable)
	if(len(sigInput)%order != 0):
		print("AMPD: Invalid order, decreasing order")
		while(len(sigInput)%order != 0):
			order -= 1
		print("AMPD: Using order " + str(order))

	N = int(len(sigInput) / order / 2)

	# Loop function calls
	for i in range(0, len(sigInput)-N, N):
		print("\t sector: " + str(i) + "|" + str((i+2*N-1)))
		pksTemp = ampd(sigInput[i:(i+2*N-1)])
		if(i == 0):
			pks = pksTemp
		else:
			pks = np.concatenate((pks, pksTemp+i))
		
	# Keep only unique values
	pks = np.unique(pks)
	
	return pks
		