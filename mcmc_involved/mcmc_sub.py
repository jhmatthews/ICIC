'''
			prelim.py 
			James Matthews

5/9/14		University of Southampton

ICIC DATA ANALYSIS WORKSHOP 2014: PRE-WORKSHOP
EXERCISES


The purpose of this preparatory exercise is to ensure you are are set up on your laptop
to do a few of the basic things which will be needed for the hands-on exercises in the
workshop. Basically this means you have some programming language installed and you
are comfortable using it (it's your choice what you use; if you want to try R, which is used
extensively in the statistics community, download from http://www.r-project.org/). You
will need to do some programming, read in data files, plot data, and draw some random
numbers from uniform and gaussian distributions; if you don't do these, you won't be
able to take advantage of the hands-on part of the workshop (or you will be spending
your time in the workshop catching up). The tasks themselves are pretty straightforward
- the theory section is here for completeness, but actually all that is needed from this
section is to be able to program a couple of the formulae (specifically equations 2-1).
'''

import sys, os
import numpy as np
import logging
import warnings
from numpy import exp
import pylab as p
from pylab import *
from astropy import units
from constants import *
from read_output import setpars
setpars()

H0_70 = 70.0 # km /s / MPC


def mu(d_l):

	'''
	distance modulus for D_L in MPC, equation (2)
	'''

	return 25.0 + 5.0 * np.log10(d_l)


# def eta(a, omega_m):

# 	'''
# 	fitting function eta, equation (3)
# 	'''

# 	scubed = (1.0 - omega_m) /  omega_m

# 	s = scubed ** (1.0/3.0)

# 	x = 1.0 / (a*a*a*a) 
# 	x -= 0.1540 * s / (a*a*a)
# 	x += 0.4304 * s * s / (a*a)
# 	x += 0.19097 * s * s * s / (a)
# 	x += 0.0066941 * s * s * s * s

# 	x = x**(-1.0/8.0)

# 	x *= 2.0 * np.sqrt(scubed + 1.0)

# 	return x


def d_l_fit(z, omega_m, H0):

	'''
	fitting function, uses eta

	For a flat Universe, we can use an accurate fitting formula, 
	given by U.-L. Pen, ApJS, 120:4950, 1999

	z is redshift
	omega_m is omega matter
	H0 is hubble constant in km/s/Mpc
	'''

	a1 = 1.0 / (1.0 + z)

	d = eta(1, omega_m) - eta(a1, omega_m)

	d *= (1.0 + z) * C / H0 / 1e5		# 1e5 converts to cm/s

	return d

def eta(a, omega_m):

	'''
	fitting function eta, equation (3)
	'''

	scubed = (1.0 - omega_m) /  omega_m

	s = scubed ** (1.0/3.0)

	x = 1.0 / (a*a*a*a) 
	x -= 0.1540 * s / (a*a*a)
	x += 0.4304 * s * s / (a*a)
	x += 0.19097 * s * s * s / (a)
	x += 0.066941 * s * s * s * s

	x = x**(-1.0/8.0)

	x *= 2.0 * np.sqrt(scubed + 1.0)

	return x



def read_SN(fname="SN.txt"):

	'''
	read supernova data

	format 
	#SN        z     mu      sigma    quality
	SN90O   0.030   35.90   0.21     Gold
	'''

	z, mu, sigma = np.loadtxt(fname, comments="#", usecols=(1,2,3), unpack=True)

	return z, mu, sigma





def question1():

	omegas = np.arange(0.2,0.6,0.1)

	zs = np.arange(0,2,0.001)

	for o in omegas:

		distances = d_l_fit(zs, o, H0_70)

		p.plot(zs, mu(distances), label="$\Omega_m=%.1f$" % o)

	p.legend(loc=4)
	p.xlabel("$z$")
	p.ylabel("$\mu$")
	p.savefig("question1.png")



def question2():

	z, mu, sigma = read_SN()

	p.errorbar(z, mu, yerr=sigma, fmt='.')
	p.xlabel("$z$")
	p.ylabel("$\mu$")
	p.savefig("question2.png")


def get_random_sn(omega_m = 0.3, rms = 0.1, n = 20):

	zs = 2.0*np.random.rand(n) 

	mu_gauss = 0.1 * np.random.randn(n)

	ds = d_l_fit(zs, omega_m, H0_70)

	mus = mu(ds)

	mus += mu_gauss

	return zs, mus

def question4():

	z, mu = get_random_sn()

	p.scatter(z,mu)

	p.savefig("question4.png")

	return z, mu

def question5(z, mu):

	p.hist(z)

	savefig("question5.png")

def do():
	clf()
	question1()
	question2()

	z, mu = question4()

	p.clf()

	question5(z, mu)

	return 0







