
def get_walker_details(NWALKERS, nproc, my_rank):
	'''
	gets details on how many samples and walkers for each
	thread, based on the total number of samples
	'''
	n_walkers = NWALKERS / nproc					# number of walkers for each thread
	remainder = NWALKERS - ( n_walkers * nproc )	# the remainder. e.g. your number of models may 

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

	return n_walkers, my_nmin, my_nmax