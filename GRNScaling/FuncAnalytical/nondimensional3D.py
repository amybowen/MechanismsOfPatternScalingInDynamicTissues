import numpy as np
import matplotlib.pyplot as plt

def find_m_ana_SS(vm,km,Dm,sourceRatioOrSize,N,L,xCurrent,ratio_given=True): 
    """ Function to find the analytical solution for the normalised morphogen concentration m at steady state at a length L """
    dy = L/(N-1) # space step
    if ratio_given == True:
        sSizeCurrent = L*sourceRatioOrSize # source is scaled to L
    else:
        sSizeCurrent = sourceRatioOrSize # source is not scaled to L, input is the sourceSize
    
    wCurrent = round(sSizeCurrent/dy)+1 # the index for the edge of the source: so the source spans from 0 to sSize0 - round this down 
    lam = np.sqrt(Dm/km)
    m_ana_i = (vm/km) * (1 + np.sinh((sSizeCurrent-L)/lam)*np.cosh(xCurrent[0:wCurrent]/lam)/np.sinh(L/lam)) # within the source
    m_ana_o = (vm/km) * (np.sinh(sSizeCurrent/lam) / np.sinh(L/lam)) * np.cosh((L-xCurrent[wCurrent:N])/lam) # outside the source
    m_ana = np.append(m_ana_i,m_ana_o)
    return m_ana

def find_m_ana_SS_npfriendly(vm,km,Dm,sourceRatioOrSize,N,L,xCurrent,ratio_given=True): 
    """ Function to find the analytical solution for the normalised morphogen concentration m at steady state at a length L """
    dy = L/(N-1) # space step
    if ratio_given == True:
        sSizeCurrent = L*sourceRatioOrSize # source is scaled to L
    else:
        sSizeCurrent = sourceRatioOrSize # source is not scaled to L, input is the sourceSize
    
    wCurrent = int(np.round(sSizeCurrent/dy))+1 # the index for the edge of the source: so the source spans from 0 to sSize0 - round this down 
    lam = np.sqrt(Dm/km)
    m_ana_i = (vm/km) * (1 + np.sinh((sSizeCurrent-L)/lam)*np.cosh(xCurrent[0:wCurrent]/lam)/np.sinh(L/lam)) # within the source
    m_ana_o = (vm/km) * (np.sinh(sSizeCurrent/lam) / np.sinh(L/lam)) * np.cosh((L-xCurrent[wCurrent:N])/lam) # outside the source
    m_ana = np.append(m_ana_i,m_ana_o)
    return m_ana


def find_m_ana_SS_single_position(vm,km,Dm,sourceRatioOrSize,L,x,ratio_given=False): # find the analytical solution for steady state at a length L
    if ratio_given == True:
        sSizeCurrent = L*sourceRatioOrSize # source is scaled to L
    else:
        sSizeCurrent = sourceRatioOrSize # source is not scaled to L, input is the sourceSize

    lam = np.sqrt(Dm/km)

    if x < sSizeCurrent: # in the source
        m_ana = (vm/km) * (1 + np.sinh((sSizeCurrent-L)/lam)*np.cosh(x/lam)/np.sinh(L/lam)) # within the source
        # print('in')
    elif x >= sSizeCurrent:
        m_ana = (vm/km) * (np.sinh(sSizeCurrent/lam) / np.sinh(L/lam)) * np.cosh((L-x)/lam) # outside the source
        # print('out')
    return m_ana

def find_L_ana_constant_growth(L0,g_max,epsilon,time): # constant growth analytical solutions. Also applies to any x(t)
    return L0*np.exp(g_max*time/(1+epsilon))


def find_t_g_constant_growth(L0,g_max,epsilon,Lf):
    return ((1+epsilon)/g_max)*np.log(Lf/L0)


def find_x_critical_from_m_critical_ana(m_critical,vm,D,k,sourceSize):
    decay_length = np.sqrt(D/k)
    if  m_critical >  (vm/k) * (1 - np.exp(-sourceSize/decay_length)): # larger than max
        return np.nan
    elif m_critical > (vm/k) * np.sinh(sourceSize/decay_length) * np.exp(-sourceSize/decay_length): # within source
        return decay_length * np.arccosh( np.exp(sourceSize/decay_length) * ( 1 - m_critical * k / vm ) )
    else:
        return - decay_length * np.log( k * m_critical / ( vm * np.sinh(sourceSize/decay_length) ) ) # outside of source

def find_x_critical_from_m_critical_anaArray(m_critical,vm,D,k,sourceSizeArray):
    decay_length = np.sqrt(D/k)
    x_criticalArray = np.zeros(len(sourceSizeArray))

    for i in range(len(sourceSizeArray)):
        sourceSize = sourceSizeArray[i]
        if  m_critical >  (vm/k) * (1 - np.exp(-sourceSize/decay_length)): # larger than max
            x_criticalArray[i] =  np.nan
        elif m_critical > (vm/k) * np.sinh(sourceSize/decay_length) * np.exp(-sourceSize/decay_length): # within source
            x_criticalArray[i] = decay_length * np.arccosh( np.exp(sourceSize/decay_length) * ( 1 - m_critical * k / vm ) )
        else:
            x_criticalArray[i] = - decay_length * np.log( k * m_critical / ( vm * np.sinh(sourceSize/decay_length) ) ) # outside of source
    
    return x_criticalArray


# %%%%%%%%%%% Analytical solutions for n1 and n2 for the linear inhibition case where p, q = 1 %%%%%%%%%%%
def find_n1_SS_ana(mCurrent,v1,v2,k2,k1,h):  # given the morphogen concentration
    """ A function to find the analytical n1 profile for the case where p = q = 1"""
    v1m = v1*(mCurrent**h/(1 + mCurrent**h))
    b = 1 + v2/(k2) - v1m/(k1)
    c = - v1m/k1
    n1_SS_p = (-b + np.sqrt(b**2 - 4*c))/(2)
    if np.all(np.less(n1_SS_p,np.zeros(len(np.array(n1_SS_p,ndmin=1))))) == False: # check that n1_SS_p is the positive root
        n1_SS = n1_SS_p
    else:
        n1_SS =  (-b - np.sqrt(b**2 - 4*c))/(2)
    return n1_SS

def find_n2_SS_ana(mCurrent,v1,v2,k2,k1,h):  # given the morphogen concentration
    """ A function to find the analytical n2 profile for the case where p = q = 1"""
    v1m = v1*(mCurrent**h/(1 + mCurrent**h))
    b = 1 + v1m/(k1) - v2/(k2)
    c = - v2/k2
    n2_SS_p = (-b + np.sqrt(b**2 - 4*c))/(2)
    if np.all(np.less(n2_SS_p,np.zeros(len(np.array(n2_SS_p,ndmin=1))))) == False: # check that n2_SS_p is the positive root
        n2_SS = n2_SS_p
    else:
        n2_SS = (-b - np.sqrt(b**2 - 4*c))/(2)
    return n2_SS



def find_gamma_bounds(Lf_values,D,k,m_critical,chi_0,L0,vm,beta,Delta=0):
    lam = np.sqrt(D/k)
    exponent = (chi_0 * Lf_values / L0)  - Delta
    return np.log( lam * np.arcsinh( k* m_critical * np.exp( exponent/lam ) / vm ) / beta ) / np.log(Lf_values)


def find_xc(L_values, D, k, vm, beta, gamma, m_critical):
    lam = np.sqrt(D/k)
    return lam * np.log( vm * np.sinh( (beta * L_values**gamma) / lam) / (k*m_critical) )

def find_gamma_max(beta,Lf):
    """ Function to find the maximum value of gamma for which the source size is less than the tissue size at final length Lf. """
    return 1 - np.log(beta)/np.log(Lf)


def find_max_L(beta,gamma):
    return beta**(- (1/(gamma-1) ) )