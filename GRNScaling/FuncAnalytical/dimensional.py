""" FuncAnalytical:
    A module containing functions for analytical equations in the case of various growth rules:
          constant growth
          homogeneous exponentially decaying growth;
          normalised temporal derivative growth, when at the critical point, in the case of constant degradation and non degradation

Includes functions for finding:
%%%%%% morphogen dynamics %%%%%%
          steady state morphogen concentrations, and derivative wrt space and amplitudes
%%%%%% growth dynamics %%%%%%
          the length of the tissue over time L(t), or the final length L*,f, and the derivatives wrt time
          the time to reach a certain length
%%%%%% patterning quantification %%%%%%
          the expectation of x, E[x]
          x_Z
%%%%%% GRN dynamics %%%%%%
          the steady state concentration profiles of the genes in the 2 node system, n1 and n2 
%%%%%% tau_sigma analysis %%%%%%
           finding tau_sigma analytical
    
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def find_M_ana_SS(vM,k,D,sourceRatioOrSize,N,L,xCurrent,ratio_given=True): # find the analytical solution for steady state at a length L
    """ Function to find the analytical solution for the morphogen concentration M at steady state at a length L """
    dy = L/(N-1) # space step
    if ratio_given == True:
        sSizeCurrent = L*sourceRatioOrSize # source is scaled to L
    else:
        sSizeCurrent = sourceRatioOrSize # source is not scaled to L, input is the sourceSize
    wCurrent = round(sSizeCurrent/dy)+1 # the index for the edge of the source: so the source spans from 0 to sSize0 - round this down 
    lam = np.sqrt(D/k)
    M_ana_i = (vM/k) * (1 + np.sinh((sSizeCurrent-L)/lam)*np.cosh(xCurrent[0:wCurrent]/lam)/np.sinh(L/lam)) # within the source
    M_ana_o = (vM/k) * (np.sinh(sSizeCurrent/lam) / np.sinh(L/lam)) * np.cosh((L-xCurrent[wCurrent:N])/lam) # outside the source
    M_ana = np.append(M_ana_i,M_ana_o)
    return M_ana

def find_M_ana_SS_lim_L(vM,k,D,sourceRatioOrSize,N,L,xCurrent,ratio_given=True): # find the analytical solution for steady state at a length L
    dy = L/(N-1) # space step
    if ratio_given == True:
        sourceSize = L*sourceRatioOrSize # source is scaled to L
    else:
        sourceSize = sourceRatioOrSize # source is not scaled to L, input is the sourceSize
    w0 = round(sourceSize/dy)+1 # the index for the edge of the source: so the source spans from 0 to sSize0 - round this down 
    decay_length = np.sqrt(D/k)
    m_ana_lim_i = (vM /k) * ( 1 - np.exp(-sourceSize/decay_length) * np.cosh(xCurrent[0:w0]/decay_length) )
    m_ana_lim_o = (vM * np.sinh(sourceSize/decay_length) /k) * np.exp(-xCurrent[w0:]/decay_length)
    M_ana = np.append(m_ana_lim_i,m_ana_lim_o)
    return M_ana

# removed N - check use of this
def find_M_ana_SS_single_position(vM,k,D,sourceRatioOrSize,L,x,ratio_given=False): # find the analytical solution for steady state at a length L
    if ratio_given == True:
        sSizeCurrent = L*sourceRatioOrSize # source is scaled to L
    else:
        sSizeCurrent = sourceRatioOrSize # source is not scaled to L, input is the sourceSize

    lam = np.sqrt(D/k)
    # check if its in the source
    if x < sSizeCurrent: # in the source
        M_ana = (vM/k) * (1 + np.sinh((sSizeCurrent-L)/lam)*np.cosh(x/lam)/np.sinh(L/lam)) # within the source
        # print('in')
    elif x >= sSizeCurrent:
        M_ana = (vM/k) * (np.sinh(sSizeCurrent/lam) / np.sinh(L/lam)) * np.cosh((L-x)/lam) # outside the source
        # print('out')
    return M_ana

def find_x_from_M_ana_SS(M_input,D,k,vM,sourceSize,L):
    lam = np.sqrt(D/k)
    M_edge = ( vM/k ) * np.sinh( sourceSize / lam) * np.cosh( (L - sourceSize)/lam ) / np.sinh( L / lam)
    if M_input > M_edge: # in source
        x_val = lam * np.arccosh( ( 1 - M_input * k / vM) * np.sinh(L/lam)/ ( np.sinh( (L-sourceSize)/lam ) ) )
    else: # out of source
        print("out of source")
        x_val = L - lam * np.arccosh( (M_input * k / vM ) * np.sinh( L / lam) / ( np.sinh( sourceSize / lam ) ) )    
    return x_val

def find_dMdx_ana_SS(vM,k,D,sRatio,N,L,xCurrent):
    lam = np.sqrt(D/k)
    dy = L/(N-1) # space step
    sSizeCurrent = L*sRatio
    wCurrent = round(sSizeCurrent/dy)+1 # the index for the edge of the source: so the source spans from 0 to sSize0 - round this down 
    dMdx_SS_i = -(1/lam) * (vM/k)* (np.sinh(( L * (1-sRatio))/lam)) * (np.sinh(xCurrent[0:wCurrent]/lam)/np.sinh(L/lam))
    dMdx_SS_o = -(1/lam) * (vM/k)* (np.sinh(( L * sRatio)/lam)) * (np.sinh( (L - xCurrent[wCurrent:N]) /lam)/np.sinh(L/lam))
    dMdx_SS = np.append(dMdx_SS_i,dMdx_SS_o)
    return dMdx_SS


def find_amplitude_SS(vM,k,sRatio,D,L):
    lam = np.sqrt(D/k)
    AmpL = (vM/k)*np.sinh(L*sRatio/lam)/np.sinh(L/lam)
    return AmpL

def find_M_ana_temp_crit(xCurrent,vM,k,g00,LCurrent,D,wCurrent,sRatio,N,beta): # g is the magnitude of growth at 0 - should be homogeneous
    k_g = k+g00*(1+beta)
    sSizeCurrent = LCurrent*sRatio
    # could put wCurrent as closer index 
    # sStep = xCurrent[wCurrent] - xCurrent[wCurrent-1]
    # sSizeError = sSizeCurrent - xCurrent[wCurrent-1]
    lam = np.sqrt(D/k_g)
    M_ana_i = (vM/k_g) * (1 - np.sinh((LCurrent-sSizeCurrent)/lam)*np.cosh(xCurrent[0:wCurrent]/lam)/np.sinh(LCurrent/lam)) # within the source
    M_ana_o = (vM/k_g) * (np.sinh(sSizeCurrent/lam) / np.sinh(LCurrent/lam)) * np.cosh((LCurrent-xCurrent[wCurrent:N])/lam) # outside the source
    M_ana = np.append(M_ana_i,M_ana_o)
    return M_ana

def find_L_ana_constant_growth(L0,g_max,epsilon,time): # constant growth analytical solutions. Also applies to any x(t)
    return L0*np.exp(g_max*time/(1+epsilon))

def find_w_ana_constant_growth(L0,g_max,epsilon,beta,gamma,time): # constant growth analytical solutions. Also applies to any x(t)
    L_ana = L0*np.exp(g_max*time/(1+epsilon))
    return beta * L_ana**gamma # source size grows with the tissue size


def find_initial_position_from_final(x_final,g_max,epsilon,time):
    return x_final * np.exp(-g_max*time / (1 + epsilon))

def find_L_ana_exp_decay_growth(L0,g_max,epsilon,alpha,time):
    return L0*np.exp((g_max/alpha)*(1-np.exp(-time*alpha))/(1+epsilon))

def find_dLdt_ana_exp_decay_growth(L0,g_max,epsilon,alpha,time):
    G = g_max/(alpha*(1+epsilon))
    return L0*G*alpha*np.exp(-alpha*time)*np.exp((g_max/alpha)*(1-np.exp(-time*alpha))/(1+epsilon))

def find_Lf_exp_decay_growth(L0,g_max,alpha,epsilon):
    return L0*np.exp(g_max/(alpha*(1+epsilon)))

def find_t_g_exp_decay_growth(L0,g_max,alpha,epsilon,Lf):
    G = g_max/(alpha*(1+epsilon))
    t_g = (-1/alpha)*np.log(1 - (1/G)*np.log(Lf/L0))
    return t_g

def find_t_g_constant_growth(L0,g_max,epsilon,Lf):
    return ((1+epsilon)/g_max)*np.log(Lf/L0)

def find_Lf_temp_deriv_growth_crit(L0,k,g00,beta_crit):
    return L0*np.sqrt(1+g00*(1+beta_crit)/k)

def find_L_ana_temp_deriv_growth_crit_0deg(time,L0,g00,epsilon):
    return L0*np.sqrt(1+2*g00*time/(1+epsilon))

def find_g0_temp_deriv_crit(time,g00,epsilon,beta_crit,k0): # find the time dependent growth for the temporal derivative growth rule with non-zero degradation
    tau = (1+beta_crit)*(1+epsilon)/(2*k0)
    return g00*np.exp(-time/tau)/(1 + (2*tau*g00*(1-np.exp(-time/tau) )/(1+epsilon) ))

def find_g0_temp_deriv_crit_0deg(time,g00,epsilon): # find the time dependent growth for the temporal derivative growth rule with zero degradation
    return g00/( 1 + 2*g00*time/(1+epsilon) )

# # advection velocity
def find_u_constant_growth(g_max,epsilon,x0,timeArray):
    return ( x0*g_max/(1+epsilon) ) * np.exp( g_max * timeArray / (1+epsilon) )

def find_sourceSize_ana_from_L(L_ana,sourceSize0,gamma):
    beta = sourceSize0 * L_ana[0] **(-gamma) # beta value for the same initial source size w0 
    return beta * L_ana**gamma

def find_sourceSizeArray(L0,g_max,epsilon,gamma,sourceSize0,timeArray):
    L_ana = find_L_ana_constant_growth(L0,g_max,epsilon,timeArray)
    beta = sourceSize0 * L0**(-gamma)
    return beta * L_ana**gamma


# %%%%%%%%%%% Analytical solutions %%%%%%%%%%%
def find_N1_SS_ana(MCurrent,V1,Mstar,N1star,V2,k2,N2star,k1,h):  # given the morphogen concentration
    """ A function to find the analytical N1 profile for the case where p = q = 1"""
    V1M = V1*(MCurrent**h/(Mstar**h + MCurrent**h))
    a = 1/N1star
    b = 1 + V2/(k2 * N2star) - V1M/(k1*N1star)
    c = - V1M/k1
    N1_SS_p = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    if np.all(np.less(N1_SS_p,np.zeros(len(np.array(N1_SS_p,ndmin=1))))) == False: # check that N1_SS_p is the positive root
        N1_SS = N1_SS_p
    else:
        N1_SS =  (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    return N1_SS

def find_N2_SS_ana(MCurrent,V1,Mstar,N1star,V2,k2,N2star,k1,h):  # given the morphogen concentration
    """ A function to find the analytical N2 profile for the case where p = q = 1"""
    V1M = V1*(MCurrent**h/(Mstar**h + MCurrent**h))
    a = 1/N2star
    b = 1 + V1M/(k1 * N1star) - V2/(k2*N2star)
    c = - V2/k2
    N2_SS_p = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    if np.all(np.less(N2_SS_p,np.zeros(len(np.array(N2_SS_p,ndmin=1))))) == False: # check that N2_SS_p is the positive root
        N2_SS = N2_SS_p
    else:
        N2_SS = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    return N2_SS

# analytical x_threshold 
def find_x_Z_SS_ana(L,w,D,k,v,Z):
    lam = np.sqrt(D/k)
    x_Z_SS_ana = L - w - lam*np.arccosh((k/v)*Z*np.sinh(L/lam)/(np.sinh(w/lam)))
    return x_Z_SS_ana

def find_x_Z_ana_lim_large_L(D,k,v,Z):
    lam = np.sqrt(D/k)
    x_Z_lim_large_L = -lam*np.log(2*Z*k/v)
    return x_Z_lim_large_L


# analytical E[x]
def find_exp_x_ana(D,k,L,sRatio):
    lam = np.sqrt(D/k)
    exp_x_ana = - lam/(np.sinh((L*(1-sRatio)/lam))) + lam/(np.tanh((L*(1-sRatio)/lam)))
    return exp_x_ana

def find_x_critical_from_M_critical_ana(M_critical,vM,D,k,sourceSize):
    decay_length = np.sqrt(D/k)
    if  M_critical >  (vM/k) * (1 - np.exp(-sourceSize/decay_length)): # larger than max
        return np.nan
    elif M_critical > (vM/k) * np.sinh(sourceSize/decay_length) * np.exp(-sourceSize/decay_length): # within source
        return decay_length * np.arccosh( np.exp(sourceSize/decay_length) * ( 1 - M_critical * k / vM ) )
    else:
        return - decay_length * np.log( k * M_critical / ( vM * np.sinh(sourceSize/decay_length) ) ) # outside of source

def find_t_sigma_ana_constant_chi(alpha,epsilon,RArray,sigma,R_value,g_value):
    # g_value = g(T); R_value = R(T) - must be at the same time point
    # sigma is the RELATIVE threshold for the error in R
    G = g_value/(alpha*(1+epsilon))
    # R_c = np.sqrt(D/k)/L0
    # t_sigma_p = (-1/alpha)*np.log((1 + (1/G)*np.log(RArray/R_value))/(1 + (1/G)*np.log((RArray*(1+sigma))/R_value)))
    # t_sigma_n = (-1/alpha)*np.log((1 + (1/G)*np.log(RArray/R_value))/(1 + (1/G)*np.log((RArray*(1-sigma))/R_value)))

    t_sigma_p = -(1/alpha)*np.log( 1 + np.log(1+sigma)/( G + np.log(RArray/R_value) ) )
    t_sigma_n = -(1/alpha)*np.log( 1 + np.log(1-sigma)/( G + np.log(RArray/R_value) ) )
    t_sigma_pos = [t_sigma_p[index]  if t_sigma_p[index] > 0 else t_sigma_n[index] for index in range(len(t_sigma_p))]
    t_sigma_neg = [t_sigma_p[index]  if t_sigma_p[index] < 0 else t_sigma_n[index] for index in range(len(t_sigma_p))]
    return np.array(t_sigma_pos),np.array(t_sigma_neg)

def find_gamma_max(beta,Lf):
    """ Function to find the maximum value of gamma for which the source size is less than the tissue size at final length Lf. """
    return 1 - np.log(beta)/np.log(Lf)