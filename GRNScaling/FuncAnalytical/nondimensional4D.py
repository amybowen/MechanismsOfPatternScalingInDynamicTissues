import numpy as np
import matplotlib.pyplot as plt

def find_m_ana_SS(vtilde,ktilde,Dtilde,sourceRatioOrSize,N,L,xCurrent,ratio_given=True): 
    """ Function to find the analytical solution for the normalised morphogen concentration m at steady state at a length L """
    dy = L/(N-1) # space step
    if ratio_given == True:
        sSizeCurrent = L*sourceRatioOrSize # source is scaled to L
    else:
        sSizeCurrent = sourceRatioOrSize # source is not scaled to L, input is the sourceSize
    wCurrent = round(sSizeCurrent/dy)+1 # the index for the edge of the source: so the source spans from 0 to sSize0 - round this down 
    lam = np.sqrt(Dtilde/ktilde)
    m_ana_i = (vtilde/ktilde) * (1 + np.sinh((sSizeCurrent-L)/lam)*np.cosh(xCurrent[0:wCurrent]/lam)/np.sinh(L/lam)) # within the source
    m_ana_o = (vtilde/ktilde) * (np.sinh(sSizeCurrent/lam) / np.sinh(L/lam)) * np.cosh((L-xCurrent[wCurrent:N])/lam) # outside the source
    m_ana = np.append(m_ana_i,m_ana_o)
    return m_ana


def find_L_ana_constant_growth(L0,gtilde,epsilon,time_nd): # constant growth analytical solutions. Also applies to any x(t)
    return L0*np.exp(gtilde*time_nd/(1+epsilon))


def find_t_nd_g_constant_growth(L0,g_nd,epsilon,Lf):
    return ((1+epsilon)/g_nd)*np.log(Lf/L0)