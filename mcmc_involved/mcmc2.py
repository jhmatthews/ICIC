#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
import pylab as p 
import numpy as np 
import math 
import scipy 
import mcmc_sub as sub
import time, sys
from mpi4py import MPI

def log_normal_gaussian(x, mu, sigma):

	num = (x - mu) ** 2 
	denom = sigma**2
	y = -0.5 * num / denom

	return np.log(1.0 / np.sqrt(2.0 * np.pi) / sigma) + y


def Log(print_string):

	if my_rank == 0:
		print print_string

	return 0

def do_general_mc (start_params, NSAMPLES, data, f_likelihood, sigmas, prior = None, param_limits=None, return_accept=False):

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

		sigmas				array-like
							sigmas for your proposal distribution
							 	

		prior 				function or NoneType
							if none use uniform prior i.e. sample likelihood, not posterior
							if function needs to take array of shape params as argument

	Returns:
		params 				array-like
							shape (nparams, NSAMPLES)
							The parameters associated 

		ps 					array-like
							shape (nparams, NSAMPLES)
							posterior probabilities for each sample					
	'''

	nparams = len(start_params)
	accept = 0

	# create blank arrays- this is where we store our walk
	params = np.zeros( (nparams, NSAMPLES) )
	ps = np.zeros(NSAMPLES)

	# set starting values and evaluate the likelihood
	params[:,0] = start_params
	ps[0] = f_likelihood(start_params, data)

	if prior != None:
		ps[0] += prior(start_params)

	t0 = time.time()

	for i in range(1,NSAMPLES):

		if (float(i) % 100) == 0:
			tnow = time.time()
			eta = (NSAMPLES - i) * (tnow - t0) / (float(i))
			Log("Number %i, time elapsed %.2fs, eta %.2fs" % (i, tnow - t0, eta) )
			Log("Acceptance %.2fpc" % (float(accept) / i))
			#print "Thread %i: %i, time elapsed %.2fs, eta %.2fs" % (my_rank, i, tnow - t0, eta)
			#sys.stdout.write("Number %i, time elapsed %.2fs, eta %.2fs" % (i, tnow - t0, eta))
			#sys.stdout.write("Acceptance %.2fpc\n" % (float(accept) / i))
			#sys.stdout.flush()


		# generate trial points from our proposal distribution
		# change this
		trial_params = np.zeros(nparams)

		for j in range(nparams):
			trial_params[j] = np.random.normal(params[:,i-1][j], sigmas[j])


		# calculate likelihood of proposed point
		ptrial = f_likelihood(trial_params, data)
		if prior != None:
			ptrial += prior(trial_params)

		# random number for test below
		z = np.random.random()

		# we use logs here to avoid numerical error
		test = np.exp(ptrial - ps[i-1])

		if z < test:
			#accept
			params[:,i] = trial_params
			ps[i] = ptrial
			accept += 1

		else:
			#copy 
			params[:,i] = params[:,i-1]
			ps[i] = ps[i - 1]

	ps = np.e ** ps

	tnew = time.time()

	if return_accept:
		return params, ps, np.float(accept) / NSAMPLES
	else:
		return params, ps


def make_plot(params, ps, fname, labels, lims, twodhist=p.scatter, nbins = 100):

	'''
	make a triangular shaped plot, with 2D probability
	distributions flanked by histograms 
	'''

	nparams = len(params)

	for i in range(nparams):

		i_hist = 1 + (i * (nparams + 1))

		p.subplot(nparams, nparams, i_hist)

		p.hist(params[i], bins=nbins, normed=True)

		n2d = i

		p.xlabel(labels[i])
		p.xlim(lims[i])

		for j in range(n2d):

			i_2d = i_hist - n2d + j

			ix = j
			iy = i

			p.subplot(nparams, nparams, i_2d)
			twodhist(params[ix], params[iy], c=ps, edgecolors="None")

			p.xlim(lims[ix])
			p.ylim(lims[iy])

			p.xlabel(labels[ix])
			p.ylabel(labels[iy])

	p.savefig(fname)
	p.clf()

def h_prior(params):

	h_mu = 0.738
	h_sigma = 0.024 
	weight = log_normal_gaussian(params[1], h_mu, h_sigma)

	return weight

def h_prior_narrow(params):

	h_mu = 0.738
	h_sigma = 0.0024 
	weight = log_normal_gaussian(params[1], h_mu, h_sigma)

	return weight

def log_likelihood_chi2(params, data):

	'''
	return the log likelihood for the omega_matter
	and little h 
	'''

	z = data[0]
	mu = data[1]
	sigma = data[2]

	omega = params[0]
	h = params[1]

	# this condition needed because proposal distribution 
	# can try to sample outside allowed param space
	if omega < 0.0 or h < 0.0 or h > 1.0 or omega > 1.0:
		return 0.0 

	dl = sub.d_l_fit(z, omega, h*100.0)

	mu_th = sub.mu(dl)

	num = ( mu - mu_th) ** 2
	denom = sigma ** 2

	l = -0.5 * np.sum ( num / denom) 

	return l

def log_likelihood_chi2_nonflat(params, data):

	'''
	return the log likelihood for the omega_matter
	and little h 
	'''

	z = data[0]
	mu = data[1]
	sigma = data[2]

	omega_m = params[0]
	omega_v = params[1]
	h = params[2]

	# this condition needed because proposal distribution 
	# can try to sample outside allowed param space
	if h < 0.0 or h > 1.0:
		return 0.0 

	dl = sub.dl_nonflat(z, h*100.0, omega_m, omega_v)

	mu_th = sub.mu(dl)

	num = ( mu - mu_th) ** 2
	denom = sigma ** 2

	l = -0.5 * np.sum ( num / denom) 

	return l

def p_accept_curve(svals = np.arange(-5,0.5,0.1)):

	p_accept = np.zeros(len(svals))
	
	for i in range(len(svals)):

		s = 10.0 ** svals[i]
		sigmas = [s,s]
		params, ps, p_accept[i] = do_general_mc (starts, NSAMPLES, data, log_likelihood_chi2, sigmas, prior=h_prior_narrow, return_accept=True)
		print s, p_accept[i]

	p.plot(svals, p_accept)
	p.xlabel(r"$\sigma_{{\rm Proposal}}$")
	p.ylabel("$P_{accept}$")

	p.savefig("paccept.png")
	p.clf()

	return 0

t0 = time.time()
nproc = MPI.COMM_WORLD.Get_size()   	# number of processes
my_rank = MPI.COMM_WORLD.Get_rank()   	# The number/rank of this process
my_node = MPI.Get_processor_name()    	# Node where this MPI process runs

NWALKERS = nproc 						# number of walkers in total									# total number of models to run	
n_walkers = NWALKERS / nproc					# number of walkers for each thread
remainder = NWALKERS - ( n_walkers * nproc )	# the remainder. e.g. your number of models may 

NSAMPLES = int(sys.argv[1])							# total number of samples 
nsamples = NSAMPLES / NWALKERS			# number of samples per walker

# little trick to spread remainder out among threads
# if say you had 19 total walkers, and 4 threads
# then n_models = 4, and you have 3 remainder
# this little loop would distribute these three 
if remainder < my_rank + 1:
	my_extra = 0
	extra_below = remainder
else:
	my_extra = 1
	extra_below = my_rank

# where to start and end your loops for each thread
my_nmin = (my_rank * n_walkers) + extra_below
my_nmax = my_nmin + n_walkers + my_extra

# total number you actually do
ndo = my_nmax - my_nmin
print "This is thread %i with walkers %i to %i, chain lengths %i" % (my_rank, my_nmin, my_nmax, nsamples)	


# set barrier so print output doesn't look muddled
# just waits for other thread
MPI.COMM_WORLD.Barrier()

data = sub.read_SN()	# read data, len 3 array


# do first MC and plot up
# params, ps = do_general_mc (starts, NSAMPLES, data, log_likelihood_chi2, sigmas)
# make_plot(params, ps, "posterior1.png", labels=["$\Omega_m$", "$h$"], lims=[(0.2,0.45), (0.6,0.7)])

# # do MC with prior and plot up
# params, ps = do_general_mc (starts, NSAMPLES, data, log_likelihood_chi2, sigmas, prior=h_prior)
# make_plot(params, ps, "posterior2_withprior.png", labels=["$\Omega_m$", "$h$"], lims=[(0.2,0.45), (0.6,0.7)])

# # do MC with narrow prior and plot up
# params, ps = do_general_mc (starts, NSAMPLES, data, log_likelihood_chi2, sigmas, prior=h_prior_narrow)
# make_plot(params, ps, "posterior3_withnarrowprior.png", labels=["$\Omega_m$", "$h$"], lims=[(0,1), (0,1)])

# p_accept_curve()
# do first MC and plot up

all_params = np.empty((0,3))
all_ps = np.empty((0))

for i in range( my_nmin, my_nmax):
	starts = np.array([0.45 + 0.1*np.random.random(), 0.9+0.1*np.random.random(), 0.65+0.1*np.random.random()])
	sigmas = [0.01, 0.01, 0.01]
	params, ps = do_general_mc (starts, nsamples, data, log_likelihood_chi2_nonflat, sigmas)

	all_params = np.concatenate((all_params, params.T))
	all_ps = np.concatenate((all_ps, ps.T))

all_params = all_params.T
all_ps = all_ps.T

Log('Waiting for threads to finish...')

# set barrier so print output doesn't look muddled
# just waits for other thread
MPI.COMM_WORLD.Barrier()



xps = MPI.COMM_WORLD.gather(all_ps, root=0)
xparams = MPI.COMM_WORLD.gather(all_params, root=0)

if my_rank == 0:
	print "gathering chains from other threads"

	all_pdata = np.zeros(NSAMPLES)
	all_paramdata = np.zeros( (3, NSAMPLES) )

	print "Data gathered." 

	for i in range(nproc):
		all_pdata[i * n_walkers * nsamples: (i+1) * n_walkers * nsamples] = xps[i]
		all_paramdata[:,i * n_walkers * nsamples: (i+1) * n_walkers * nsamples] = xparams[i]

	print "Shape of param array ", all_paramdata.shape
	make_plot(params, ps, "posterior_NSAMPLES%i.png" % NSAMPLES, labels=["$\Omega_M$", "$\Omega_V$", "$h$"], lims=[(0.3,0.7), (0.7,1.5), (0.6,0.7)])


print "Total Time Elapsed %.2fs" % (time.time() - t0)
MPI.COMM_WORLD.Finalize()



