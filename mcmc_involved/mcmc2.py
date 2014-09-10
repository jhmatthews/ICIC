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

def do_general_mc (start_params, NSAMPLES, data, f_likelihood, sigmas, param_limits=None):

	'''
	Do the MCMC chain in 2-d parameter space. 
	Start at omega_start and h_start

	Args:
		params 				array-like
							len nparams
							starting parameters of your model

		NSAMPLES 			int
							number of steps in mc 

		f_likelihood		function
							likelihood function

		proposal_cov		array-like
							covariance matrix for your proposal distribution
							shape 	

	Returns:
		params 				array-like
							shape (nparams, NSAMPLES)
							The parameters associated 

		ps 					array-like
							shape (nparams, NSAMPLES)
							posterior probabilities for each sample					
	'''

	nparams = len(start_params)

	# create blank arrays- this is where we store our walk
	params = np.zeros( (nparams, NSAMPLES) )
	ps = np.zeros(NSAMPLES)

	# set starting values and evaluate the likelihood
	params[:,0] = start_params
	ps[0] = f_likelihood(start_params, data)

	for i in range(1,NSAMPLES):

		# generate trial points from our proposal distribution
		# change this
		trial_params = np.zeros(nparams)
		for j in range(nparams):
			trial_params[j] = np.random.normal(params[:,i-1][j], sigmas[j])


		# calculate likelihood of proposed point
		ptrial = f_likelihood(trial_params, data)

		# random number for test below
		z = np.random.random()

		# we use logs here to avoid numerical error
		test = np.exp(ptrial - ps[i-1])

		if z < test:
			#accept
			params[:,i] = trial_params
			ps[i] = ptrial

		else:
			#copy 
			params[:,i] = params[:,i-1]
			ps[i] = ps[i - 1]

	return params, ps




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
	params = [omegas[0], hs[0]]
	ps[0] = log_likelihood_chi2(params, data)

	for i in range(1,NSAMPLES):

		# generate trial points from our proposal distribution
		# change this
		omega_trial = np.random.normal(omegas[i-1], sigma)
		h_trial = np.random.normal(hs[i-1], sigma)

		# calculate likelihood of proposed point
		params = [omega_trial, h_trial]
		ptrial = log_likelihood_chi2(params, data)

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

def make_plot(params, ps, nbins = 100):

	nparams = len(params)

	for i in range(nparams):

		i_hist = 1 + (i * (nparams + 1))

		p.subplot(nparams, nparams, i_hist)

		p.hist(params[i], bins=nbins, normed=True)

		n2d = i

		for j in range(n2d):

			i_2d = i_hist - n2d + j

			print i_2d

			p.subplot(nparams, nparams, i_2d)
			p.scatter(params[i-1], params[i], c=ps)

	p.show()



def log_likelihood_chi2(params, data):

	z = data[0]
	mu = data[1]
	sigma = data[2]

	omega = params[0]
	h = params[1]

	if omega < 0.0 or h < 0.0 or h > 1.0 or omega > 1.0:
		return 0.0 

	dl = sub.d_l_fit(z, omega, h*100.0)

	mu_th = sub.mu(dl)

	num = ( mu - mu_th) ** 2
	denom = sigma ** 2

	l = -0.5 * np.sum ( num / denom) 

	return l



data = sub.read_SN()	# read data, len 3 array

omega_start = 0.5
h_start = 0.5
NSAMPLES = 10000
sigmas = [0.01,0.01]
proposal_cov = np.ndarray((2,2))
proposal_cov[:] = 0.1
print proposal_cov
starts = np.array([omega_start, h_start])

params, ps = do_general_mc (starts, NSAMPLES, data, log_likelihood_chi2, sigmas)
#o, h, ps = do_mc(0.5, 0.5, 10000, 0.01, data)

#p.subplot(221)
# p.scatter(params[0], params[1], c=np.e**ps)
# #p.scatter(o, h, c=ps)
# #p.hist2d(o,h, bins=100)

# #p.subplot(222)
# #p.hist(omegas)
# p.show()
make_plot(params, np.e**ps)

# zth = np.arange(0,2,0.01)
# dl = sub.d_l_fit(zth, 0.24, 0.71*100.0)
# dl2 = sub.d_l_fit(zth, 0.45, 0.63*100.0)
# mu_th = sub.mu(dl)
# mu2 = sub.mu(dl2)

# p.scatter(data[0], data[1])
# p.plot(zth, mu_th)
# p.plot(zth, mu2)
# p.show()


