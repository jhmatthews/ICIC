#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
import pylab as p 
import numpy as np 
import math 
import scipy 


def p_S(S):

	'''
	probability of S given alpha
	power law distribution of fluxes beyond some known limit S0
	alpha > 1
	'''
	alpha = 1.5
	S0 = 5.0
	norm = (alpha - 1.0) * (S0 ** (alpha - 1.0))

	return norm * (S**(-alpha))


def reject_sample(prob, low_lim = 0.0, upper_lim = 1.0, ynorm = 1.0):

	'''
	use the rejection method to sample the probability distribution prob 

	return an array with shape N
	'''
	keep_going = True

	while keep_going:
		xtest = low_lim + (np.random.random() * (upper_lim - low_lim))
		ytest = np.random.random()

		if ytest < prob(xtest):
			keep_going = False

	return xtest

N = 1000
S0 = 5.0
low_lim = S0
upper_lim = 500.0
ynorm = 1.0

y = np.array([reject_sample(p_S, low_lim = S0, upper_lim = upper_lim, ynorm = ynorm) for i in range(N)])


p.hist(y, normed = True, bins = 100)

xx = np.arange(low_lim, upper_lim, 0.01)
p.plot(xx, p_S(xx))
p.show()


