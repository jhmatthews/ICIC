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
	alpha = 2
	S0 = 0.1
	norm = (alpha - 1.0) * (S0 ** (alpha - 1.0))

	return norm * (S**(-alpha))


def reject_sample(prob, low_lim = 0.0, upper_lim = 1.0):

	'''
	use the rejection method to sample the normalised probability distribution prob 

	return an array with shape N
	'''
	keep_going = True

	norm = prob(low_lim)

	while keep_going:
		xtest = low_lim + (np.random.random() * (upper_lim - low_lim))
		ytest = norm * np.random.random()

		if ytest < prob(xtest):
			keep_going = False

	return xtest


def p_uniform(x, k, low = -1.0, up = 1.0):

	i_bool = (x > low) * (x < up)
	return k * i_bool

\




def do_q1():
	N = 1000
	S0 = 0.1
	low_lim = S0
	upper_lim = 100.0
	ynorm = 1.0

	y = np.array([reject_sample(p_S, low_lim = S0, upper_lim = upper_lim) for i in range(N)])
	p.hist(y, normed = True, bins = 1000)

	x = np.sum(y) / N

	print x

	print np.mean(y)
	print np.var(y)

	# scatter(np.arange(N), y)
	xx = np.arange(low_lim, upper_lim, 0.01)
	p.plot(xx, p_S(xx))
	p.show()


def do_q2():

	z_u = 
	z_v = 






