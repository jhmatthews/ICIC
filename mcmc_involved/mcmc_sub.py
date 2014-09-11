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
from astropy import units
from constants import *
from read_output import setpars
from scipy import integrate
setpars()

H0_70 = 70.0 # km /s / MPC


def mu(d_l):

	'''
	distance modulus for D_L in MPC, equation (2)
	'''

	return 25.0 + 5.0 * np.log10(d_l)


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




def rintegral(z, omega_m, omega_v):

	'''
	integrate to find r(z)
	this is used to calculate transverse comoving distance
	'''

	omega = omega_m + omega_v
	integral = integrate.quad(rintegrand, 0.0, z, args=(omega_m, omega_v))

	integral = integral[0]

	if omega != 1:
		integral *= np.sqrt( np.fabs(1.0 - omega))

	return integral



def rintegrand(z, omega_m, omega_v):

	'''
	The integrand 
	'''

	omega = omega_m + omega_v

	sqterm = omega_m * (1.0 + z)**3
	sqterm += omega_v
	sqterm += (1. - omega) * (1.0 + z) * (1.0 + z)

	denom = np.sqrt(sqterm)
	integrand = 1.0 / denom 

	return integrand


def S_k(r, omega):

	'''
	transverse distance from cosmology
	'''

	if omega > 1:
		return np.sin(r)
	elif omega < 1:
		return np.sinh(r)
	elif omega == 1:
		return r
	else:
		print "Don't understand omega"
		sys.exit(0)


def dl_nonflat(z, H0, omega_m, omega_v):

	'''
	Calculate dl in Mpc in a non flat universe 
	calculate dls for array of zs, for 
	constant H0, omega_m and omega_v
	'''

	omega = omega_m + omega_v

	dl = np.zeros(len(z))

	for i in range(len(z)):

		d = (1.0 + z[i]) * C / H0 / 1e5

		if omega != 1:
			d /= np.sqrt(np.fabs(1. - omega))

		r = rintegral(z[i], omega_m, omega_v)	

		s = S_k (r, omega)

		d *= s

		dl[i] = d 

	return dl



