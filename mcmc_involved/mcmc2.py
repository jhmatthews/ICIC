#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
import pylab as p 
import numpy as np 
import math 
import scipy 
import mcmc_sub as sub


def step(x):

	return (x > 0.0)

def prob(x):

	mean = 0.3
	sigma = 0.1

	exp_term = -0.5 * ( ( x - mean) / sigma ) ** 2 
	x1 = np.exp(exp_term) 
	x2 = step(x) * np.exp(- x / 0.2 )
	x3 = step(x - 0.8) * step(0.9 - x)

	return x1 + x2 + x3

def do_general_mc (params, NSAMPLES, data, f_likelihood, sigma_proposals):

	'''
	Do the MCMC chain in 2-d parameter space. 
	Start at omega_start and h_start

	:INPUTS:
		params 				array-like
							len nparams
							the parameters of your model

		NSAMPLES 			int
							number of steps in mc 

		f_likelihood		function
							likelihood function

		sigma_proposals		array-like
							sigmas for each param for your proposal distribution



	'''


def do_mc(omega_start, h_start, NSAMPLES, sigma, data):

	'''
	Do the MCMC chain in 2-d parameter space. 
	Start at omega_start and h_start
	'''

	# create blank arrays- this is where we store our walk
	omegas = np.zeros(NSAMPLES)
	hs = np.zeros(NSAMPLES)
	ps = np.zeros(NSAMPLES)

	# set starting values and evaluate the likelihood
	omegas[0] = omega_start
	hs[0] = h_start
	ps[0] = log_likelihood_chi2(omegas[0], hs[0], data)

	for i in range(1,NSAMPLES):

		# generate trial points from our proposal distribution
		# change this
		omega_trial = np.random.normal(omegas[i-1], sigma)
		h_trial = np.random.normal(hs[i-1], sigma)

		# calculate likelihood of proposed point
		ptrial = log_likelihood_chi2(omega_trial, h_trial, data)

		# random number for test below
		z = np.random.random()

		# we use logs here to avoid numerical error
		test = np.exp(ptrial - ps[i-1])

		if z < test:
			#accept
			omegas[i] = omega_trial
			hs[i] = h_trial
			ps[i] = ptrial

		else:
			#copy 
			omegas[i] = omegas[i - 1]
			hs[i] = hs[i - 1]
			ps[i] = ps[i - 1]

		#print omegas[i], hs[i], ps[i], test

	return omegas, hs, ps


def log_likelihood_chi2(omega, h, data):

	if omega < 0.0 or h < 0.0 or h > 1.0 or omega > 1.0:
		return 0.0 

	z = data[0]
	mu = data[1]
	sigma = data[2]

	dl = sub.d_l_fit(z, omega, h*100.0)
	#dlstar = sub.d_l_fit(z, omega, 100.0)

	mu_th = sub.mu(dl)

	num = ( mu - mu_th) ** 2
	denom = sigma ** 2

	l = -0.5 * np.sum ( num / denom) 

	return l



data = sub.read_SN()	# read data, len 3 array

omega_start = 0.5
h_start = 0.5
NSAMPLES = 10000
sigma = 0.01

omegas, hs, ps = do_mc(omega_start, h_start, NSAMPLES, sigma, data)

#p.subplot(221)
p.scatter(omegas, hs, c=ps)

#p.subplot(222)
#p.hist(omegas)
p.show()

# zth = np.arange(0,2,0.01)
# dl = sub.d_l_fit(zth, 0.24, 0.71*100.0)
# dl2 = sub.d_l_fit(zth, 0.45, 0.63*100.0)
# mu_th = sub.mu(dl)
# mu2 = sub.mu(dl2)

# p.scatter(data[0], data[1])
# p.plot(zth, mu_th)
# p.plot(zth, mu2)
# p.show()


