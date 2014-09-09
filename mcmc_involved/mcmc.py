#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
import pylab as p 
import numpy as np 
import math 
import scipy 


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

def do_mc(xstart, NSAMPLES, sigma):

	xs = np.zeros(NSAMPLES)
	ps = np.zeros(NSAMPLES)
	dps = np.zeros(NSAMPLES)

	xs[0] = xstart
	ps[0] = prob(xs[0])

	for i in range(1,NSAMPLES):

		x_s_minus_1 = xs[i - 1]
		p_s_minus_1 = ps[i - 1]

		xtrial = np.random.normal(xs[i-1], sigma)

		ptrial = prob(xtrial)

		z = np.random.random()

		if z < ptrial / ps[i - 1]:
			#accept
			xs[i] = xtrial
			ps[i] = ptrial

		else:
			#copy 
			xs[i] = xs[i - 1]
			ps[i] = ps[i - 1]

	return xs, ps

NSAMPLES = 10000

xs, ps = do_mc(1.0, NSAMPLES, 1.0)


p.subplot(411)
x = np.arange(-2,2,0.01)
p.plot(x, prob(x))
p.xlim(-1,2)

p.subplot(412)
p.scatter(xs, ps)
p.xlim(-1,2)

p.subplot(413)
p.scatter(np.arange(NSAMPLES), xs)


p.subplot(414)
p.hist(xs, normed = True, bins = 40)
p.xlim(-1,2)
p.show()



