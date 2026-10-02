# a module defining distribution functions in order to fit them

import numpy as np

# functions for fitting 
def straight_line(x,k,c):
    return k*x + c

def straight_line_in_log(x,A,B):
    return A*np.exp(B*x)

def quadratic_line(x,a,b,c):
    return a*(x**2) + b*x + c

def completed_square_quadratic_line(x,a,x_a,c):
    return a*((x-x_a)**2) + c

def cubic_line(x,a,b,c,d):
    return a*(x**3) + b*(x**2) + c*x + d

def morph_dist(x,c0,lam):
    return c0*np.exp(-x/lam)

def morph_dist_double_exp(x,A,B,lam):
    return A*np.exp(-x/lam) + B*np.exp(x/lam)

def double_exp(x,A,B,lam1,lam2):
    return A*np.exp(x/lam1) + B*np.exp(x/lam2)

def cosh_func(r,c0,phi):
    return c0*np.cosh((1-r)/phi)

def Gaussian_general(x, mu, sigma):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x - mu)/sigma)**2)

def Gaussian_diffusion(x,D,t):
    return 1/np.sqrt(np.pi*D*t)*np.exp(-x**2/(4*D*t))

def negative_exponential(x,a,b): # constant growth analytical solutions
    return a*(1-np.exp(-x/b))