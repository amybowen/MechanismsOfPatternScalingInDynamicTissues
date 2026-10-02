""" A set of function relating to the two node, mutual inhibitory toggle switch induced by a morphogen - using non-dimensionalised parameters m, n1, n2
The node activated by the morphogen is referred to as n1, and the other node as n2 (in non-dimensionalised parameters).
The combination of activation by morphogen and inhibition on n1 is multiplicative Hill functions defined by a Hill coefficient and a critical concentration. """

import numpy as np
import numba
import math
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
# import FuncAnalytical

""" Variables:
# X is the normalised concentration of n1, X=n1/n1star

# Parameters
# n1star is the critical Hill concentration of n1 for inhibition of n2
# n2star is the critical Hill concentration of n2 for inhibition of n1
# Mstar is the critical Hill concentration of morphogen M for activation of n1
# p is the Hill coefficient for inhibition of n1 by n2
# q is the Hill coefficient for inhibition of n2 by n1
# h is the Hill coefficient for activation of n1 by M

# Composite parameters:
# c1 = ( v1 / ( k1 * n1star ) ) * ( M / Mstar )^h / ( 1 + ( M / Mstar )^h )  
# alpha1 = v1 / ( k1 * n1star )
# alpha2 = v2 / ( k1 * n2star )
# c2 = v2 / ( k2 * n2star )
# 
# 
#  """


# %%%%%%%%%%%%%%%%%%% SS equations for n1 and n2 %%%%%%%%%%%%%%%%%%%

def find_n1_SS_given_n2(n2_values,v1,k1,m_val,h,p):
    return (v1 / k1) * ( m_val**(h) / ( 1 + m_val**(h) ) ) * (1 / (1 + (n2_values)**p ) )

def find_n2_SS_given_n1(n1_values,v2,k2,q):
    return (v2 / k2) / (1 + (n1_values)**q ) 
# %%%%%%%%%%%%%%%%%%% dni/dt equations for n1 and n2 %%%%%%%%%%%%%%%%%%%

def find_dn1dt(n1_values,n2_values,v1,k1,m_val,h,p):
    return  v1 * ( m_val**(h) / ( 1 + m_val**(h) ) ) * (1 / (1 + (n2_values)**p ) ) - k1 * n1_values

def find_dn2dt(n1_values,n2_values,v2,k2,q):
    return (v2 / (1 + (n1_values)**q ) ) - k2*n2_values


# %%%%%%%%%%%%%%%%%%% 1D solvers for n1 and n2 %%%%%%%%%%%%%%%%%%%
@numba.njit
def n1n2m_solver_NoGrowth_fixedm(N,dt,v1,v2,k1,k2,p,h,q,store_no,n1_init:np.ndarray,n2_init:np.ndarray,m_init:np.ndarray,count_max):
    """ A solver for n1 and n2 (mutual inhibitory toggle switch) with a fixed normalised morphogen profile m over time with no growth.
        Essentially a set of 0D solvers. """
    # dt is dimensional time step
    
    # initiate initial conditions
    mCurrent = np.zeros(N)
    n1Current = np.zeros(N)
    n2Current = np.zeros(N)

    # initiate previous arrays for n1 and n2
    n2Prev = np.zeros(N)
    n1Prev = np.zeros(N)

    # set the current Arrays as the initial conditions
    mCurrent[:] = m_init
    n1Current[:] = n1_init[:]
    n2Current[:] = n2_init[:]

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)

    mArray =  np.zeros((no_of_stores,N))
    n1Array = np.zeros((no_of_stores,N))
    n2Array = np.zeros((no_of_stores,N))

    # set the values to the initial values
    mArray[0,:] = m_init
    n1Array[0,:] = n1_init[:]
    n2Array[0,:] = n2_init[:]

    for count in range(1,count_max):
        # mPrev[:] = Mcurrent[:]
        n1Prev[:] = n1Current[:]
        n2Prev[:] = n2Current[:]

        for i in range(N): # iterate across space for gene products
            n1Current[i] = n1Prev[i]*(1-k1*dt) + v1*dt*((mCurrent[i])**(h)/(1 + (mCurrent[i])**(h) ) ) * (1 / (1 + (n2Prev[i])**p ) )
            n2Current[i] = n2Prev[i]*(1-k2*dt) + v2*dt / (1 + (n1Prev[i])**q )

        
        # store solutions every store_no
        if count % store_no == 0:
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            store_index = int(count/store_no)
            mArray[store_index,:] = mCurrent[:]
            n1Array[store_index,:] = n1Current[:]
            n2Array[store_index,:] = n2Current[:]


    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, mArray, n1Array, n2Array


# growing solver that doesn't scale the source
@numba.njit
def n1n2m_solver_ConstantGrowth_fixed_source(N,N_1,N_2,wCurrent,Dm,km,vm,epsilon,sourceSize,dt,dy,v1,v2,k1,k2,h,p,q,L0,store_no,m_init:np.ndarray,n1_init:np.ndarray,n2_init:np.ndarray,count_max,g):
    # ndFuncToggle.n1n2m_solver_ConstantGrowth_fixed_source(N,N_1,N_2,w0,D,k,vm,epsilon,sourceSize,dt,dy,v1,v2,k1,k2,h,p,q,L0,store_no,m_init,n1_init,n2_init,count_max,g_max)
    """ A solver for M, n1 and n2 over time with constant growth. The morphogen source size is fixed. """ 

    # initialise Current and Prev arrays for M, n1, n2, x_Y
    mCurrent = np.zeros(N)
    n1Current = np.zeros(N)
    n2Current = np.zeros(N)

    mPrev  = np.zeros(N)
    n2Prev = np.zeros(N)
    n1Prev = np.zeros(N)
    x_YPrev = np.zeros(N)

    # set initial conditions as current arrays
    xCurrent = np.linspace(0,L0,N) #The co-moving coordinates: N spatial points, includes the point 0 and L0
    mCurrent[:] = m_init[:]
    n1Current[:] = n1_init[:]
    n2Current[:] = n2_init[:]
    x_YCurrent = np.ones(N) # initial x_Y
    prodRat = 0 # production ratio between the last cell in the source and first cell out, to interpolate

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)
    # 2D
    mArray =  np.zeros((no_of_stores,N))
    n1Array = np.zeros((no_of_stores,N))
    n2Array = np.zeros((no_of_stores,N))
    xArray = np.zeros((no_of_stores,N))
    x_YArray = np.zeros((no_of_stores,N))
    # 1D
    LArray = np.zeros(no_of_stores)
    ssArray = np.zeros(no_of_stores)

    # set the values to the initial values
    mArray[0,:] =  m_init[:]
    n1Array[0,:] = n1_init[:]
    n2Array[0,:] = n2_init[:]
    xArray[0,:] = xCurrent[:]
    x_YArray[0,:] = x_YCurrent[:]
    LArray[0] = L0
    ssArray[0] = sourceSize

    for count in range(1,count_max):
        mPrev[:] = mCurrent[:]
        n1Prev[:] = n1Current[:]
        n2Prev[:] = n2Current[:]
        x_YPrev[:] = x_YCurrent[:]

        # for loop to update the morphogen
        # within source
        for i in range(1,wCurrent): # iterate across space to the point before wCurrent
            mCurrent[i] = mPrev[i]*(1-(km+g)*dt) + vm*dt + dt*( ( 2*Dm / (dy**2) ) * (1/x_YPrev[i]) * ( ( (mPrev[i+1] - mPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (mPrev[i] - mPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # at the index wCurrent, which has a production prodRat
        mCurrent[wCurrent] = mPrev[wCurrent]*(1-(km+g)*dt) + prodRat*vm*dt + dt*( ( 2*Dm / (dy**2) ) * (1/x_YPrev[wCurrent]) * ( ( (mPrev[wCurrent+1] - mPrev[wCurrent])/(x_YPrev[wCurrent+1] + x_YPrev[wCurrent]) ) - ( (mPrev[wCurrent] - mPrev[wCurrent-1])/(x_YPrev[wCurrent-1] + x_YPrev[wCurrent]) ) ) )
        # outside of source

        for i in range(wCurrent+1,N_1):
            mCurrent[i] = mPrev[i]*(1-(km+g)*dt) + dt*( ( 2*Dm / (dy**2) ) * (1/x_YPrev[i]) * ( ( (mPrev[i+1] - mPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (mPrev[i] - mPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        mCurrent[0] = mPrev[0]*(1-(km+g)*dt) + vm*dt + (Dm*dt/(dy**2))*(mPrev[1] - mPrev[0] )
        mCurrent[N_1] = mPrev[N_1]*(1-(km+g)*dt) + (Dm*dt/(dy**2))*(mPrev[N_2] - mPrev[N_1] )

        # update x_Y and xArray
        Sx_Y = 0 # initiate Sum(x_Y) over Y, the value of x_Y at x=0 (Y=0) for all time
        for i in range(N): # iterate across space for gene products
            n1Current[i] = n1Prev[i]*(1-dt*(k1+g) ) + v1*dt*( (mCurrent[i])**(h) / ( 1 + (mCurrent[i])**(h) ) ) * (1 / (1 + (n2Prev[i])**p ) )
            n2Current[i] = n2Prev[i]*(1-dt*(k2+g) ) + v2*dt / (1 + (n1Prev[i])**q )
            x_YCurrent[i] = x_YPrev[i]/(1-g*dt/(1+epsilon))
            xCurrent[i] = dy*Sx_Y
            Sx_Y += x_YCurrent[i]
    

        # update length of tissue L
        LCurrent = xCurrent[N_1]
      
        # update source index in co-moving coordinates to keep the width the same
        # [wCurrent,sCurrent] = FuncUpdate.update_source(sRatio,N,LCurrent,xCurrent,sCurrent)
        wCurrent = np.searchsorted(xCurrent,sourceSize) # w is the index that sSizeIdeal should be placed in xCurrent to be in order. First index larger than sourcesize
        sStep = xCurrent[wCurrent] - xCurrent[wCurrent-1]
        sSizeError = sourceSize - xCurrent[wCurrent-1] # find the error in source size due to discretisation
        prodRat = sSizeError/sStep # the normalised fractional error in the source position
        

        # # output solutions every store_no
        if count % store_no == 0:
            store_index = int(count/store_no)
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            mArray[store_index,:] = mCurrent[:]
            n1Array[store_index,:] = n1Current[:]
            n2Array[store_index,:] = n2Current[:]
            xArray[store_index,:] = xCurrent[:]
            x_YArray[store_index,:] = x_YCurrent[:]
            LArray[store_index] = LCurrent
            ssArray[store_index] = sourceSize

    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, xArray, mArray, n1Array, n2Array, x_YArray, LArray, ssArray

# growing solve that scales the source
@numba.njit
def n1n2m_solver_ConstantGrowth_scale_source(N,N_1,N_2,wCurrent,Dm,km,vm,epsilon,beta,gamma,dt,dy,v1,v2,k1,k2,h,p,q,L0,store_no,m_init:np.ndarray,n1_init:np.ndarray,n2_init:np.ndarray,count_max,g):  
    # a loop to solve for C, N1, N2

    # initialise Current and Prev arrays for M, N1, N2, x_Y
    mCurrent = np.zeros(N)
    n1Current = np.zeros(N)
    n2Current = np.zeros(N)

    mPrev  = np.zeros(N)
    n2Prev = np.zeros(N)
    n1Prev = np.zeros(N)
    x_YPrev = np.zeros(N)

    # set initial conditions as current arrays
    xCurrent = np.linspace(0,L0,N) #The co-moving coordinates: N spatial points, includes the point 0 and L0
    mCurrent[:] = m_init[:]
    n1Current[:] = n1_init[:]
    n2Current[:] = n2_init[:]
    x_YCurrent = np.ones(N) # initial x_Y
    prodRat = 0

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)
    # 2D
    mArray =  np.zeros((no_of_stores,N))
    n1Array = np.zeros((no_of_stores,N))
    n2Array = np.zeros((no_of_stores,N))
    xArray = np.zeros((no_of_stores,N))
    x_YArray = np.zeros((no_of_stores,N))
    # 1D
    LArray = np.zeros(no_of_stores)
    ssArray = np.zeros(no_of_stores)
    # sourceSizeArray = np.zeros(no_of_stores)


    # set the values to the initial values
    mArray[0,:] =  m_init[:]
    n1Array[0,:] = n1_init[:]
    n2Array[0,:] = n2_init[:]
    xArray[0,:] = xCurrent[:]
    x_YArray[0,:] = x_YCurrent[:]
    LArray[0] = L0
    ssArray[0] = beta * L0**gamma

    for count in range(1,count_max):
        mPrev[:] = mCurrent[:]
        n1Prev[:] = n1Current[:]
        n2Prev[:] = n2Current[:]
        x_YPrev[:] = x_YCurrent[:]

        # for loop to update the morphogen
        # # within source
        for i in range(1,wCurrent): # iterate across space
            mCurrent[i] = mPrev[i]*(1-(km+g)*dt) + vm*dt + dt*( ( 2*Dm / (dy**2) ) * (1/x_YPrev[i]) * ( ( (mPrev[i+1] - mPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (mPrev[i] - mPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        

        # %%%%%%%%%%%%%%%%%%%%% up to here %%%%%%%%%%%%%%%%%%%%%
        # at in between point - taking into account the production ratio
        mCurrent[wCurrent] = mPrev[wCurrent]*(1-(km+g)*dt) + prodRat*vm*dt + dt*( ( 2*Dm / (dy**2) ) * (1/x_YPrev[wCurrent]) * ( ( (mPrev[wCurrent+1] - mPrev[wCurrent])/(x_YPrev[wCurrent+1] + x_YPrev[wCurrent]) ) - ( (mPrev[wCurrent] - mPrev[wCurrent-1])/(x_YPrev[wCurrent-1] + x_YPrev[wCurrent]) ) ) )
        # outside of source
        for i in range(wCurrent+1,N_1):
            mCurrent[i] = mPrev[i]*(1-(km+g)*dt) + dt*( ( 2*Dm / (dy**2) ) * (1/x_YPrev[i]) * ( ( (mPrev[i+1] - mPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (mPrev[i] - mPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        mCurrent[0] = mPrev[0]*(1-(km+g)*dt) + vm*dt + (Dm*dt/(dy**2))*(mPrev[1] - mPrev[0] )
        mCurrent[N_1] = mPrev[N_1]*(1-(km+g)*dt) + (Dm*dt/(dy**2))*(mPrev[N_2] - mPrev[N_1] )

        Sx_Y = 0 # initiate Sum(x_Y) over Y, the value of x_Y at x=0 (Y=0) for all time
        for i in range(N): # iterate across space for gene products
            n1Current[i] = n1Prev[i]*(1-k1*dt) + v1*dt*(mCurrent[i]**h/(1 + mCurrent[i]**h)) * (1 / (1 + (n2Prev[i])**p ) ) - dt*g*n1Prev[i]
            n2Current[i] = n2Prev[i]*(1-k2*dt) + v2*dt / (1 + (n1Prev[i])**q ) - dt*g*n2Prev[i]
            x_YCurrent[i] = x_YPrev[i]/(1-g*dt/(1+epsilon))
            xCurrent[i] = dy*Sx_Y
            Sx_Y += x_YCurrent[i]
    
        # update length of tissue L
        LCurrent = xCurrent[N_1]
        
        # update source width and corresponding index in co-moving coordinates
        # [wCurrent,sCurrent] = FuncUpdate.update_source(sRatio,N,LCurrent,xCurrent,sCurrent)
        sSizeIdeal = beta * LCurrent**gamma # the size of the source in x-space
        wCurrent = np.searchsorted(xCurrent,sSizeIdeal) # w is the index that sSizeIdeal should be placed in xCurrent to be in order
        sStep = xCurrent[wCurrent] - xCurrent[wCurrent-1]
        sSizeError = sSizeIdeal - xCurrent[wCurrent-1] # find the error in source size due to discretisation
        prodRat = sSizeError/sStep # the normalised fractional error in the source position
        
        # output solutions every store_no
        if count % store_no == 0:
            store_index = int(count/store_no)
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            mArray[store_index,:] = mCurrent[:]
            n1Array[store_index,:] = n1Current[:]
            n2Array[store_index,:] = n2Current[:]
            xArray[store_index,:] = xCurrent[:]
            x_YArray[store_index,:] = x_YCurrent[:]
            LArray[store_index] = LCurrent
            ssArray[store_index] = sSizeIdeal

    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, xArray, mArray, n1Array, n2Array, x_YArray, LArray, ssArray


# %%%%%%%%%%%%%%%%%%% 0D solvers for n1 and n2 %%%%%%%%%%%%%%%%%%%
@numba.njit(fastmath=True)
def n1n2_0D_solver(mtildeArray,v1,v2,k1,k2,h,p,q,dt,n1_init,n2_init,SS_factor=1E-6):
    SS_check_n1 = False
    SS_check_n2 = False
    N_time = len(mtildeArray) # the number of time points in solution
    # initial conditions
    # initiate solution arrays 
    n1Array = np.zeros(N_time)
    n2Array = np.zeros(N_time)

    # set initial values
    n1Array[0] = n1_init
    n2Array[0] = n2_init
    for i in range(1,N_time):
        n1Array[i] = n1Array[i-1]*(1-k1*dt) + v1*dt * ( mtildeArray[i-1]**(h)/(1 + mtildeArray[i-1]**(h) )) * (1 / (1 + (n2Array[i-1])**p ) )
        n2Array[i] = n2Array[i-1]*(1-k2*dt) + v2*dt/(1 + (n1Array[i-1])**q ) 


    if (n1Array[N_time-1] - n1Array[N_time-2]) < n1Array[N_time-1]*dt*SS_factor:
        SS_check_n1 = True

    if (n2Array[N_time-1] - n2Array[N_time-2]) < n2Array[N_time-1]*dt*SS_factor:
        SS_check_n2 = True
    
    # N1Array, N2Array, timeArray, SS_check_n1, SS_check_n2
    return n1Array, n2Array, np.arange(N_time)*dt, SS_check_n1, SS_check_n2

@numba.njit(fastmath=True)
def n1n2_0D_Growth_solver(mtildeArray,gArray_at_x,v1,v2,k1,k2,h,p,q,dt,n1_init,n2_init,SS_factor=1E-6):
    SS_check_n1 = False
    SS_check_n2 = False
    N_time = len(mtildeArray) # the number of time points in solution
    # initial conditions
    # initiate solution arrays 
    n1Array = np.zeros(N_time)
    n2Array = np.zeros(N_time)

    # set initial values
    n1Array[0] = n1_init
    n2Array[0] = n2_init
    for i in range(1,N_time):
        n1Array[i] = n1Array[i-1]*(1-(k1+gArray_at_x[i-1])*dt) + v1*dt * ( mtildeArray[i-1]**(h)/(1 + mtildeArray[i-1]**(h) )) * (1 / (1 + (n2Array[i-1])**p ) )
        n2Array[i] = n2Array[i-1]*(1-(k2+gArray_at_x[i-1])*dt) + v2*dt/(1 + (n1Array[i-1])**q ) 

    if (n1Array[N_time-1] - n1Array[N_time-2]) < n1Array[N_time-1]*dt*SS_factor:
        SS_check_n1 = True

    if (n2Array[N_time-1] - n2Array[N_time-2]) < n2Array[N_time-1]*dt*SS_factor:
        SS_check_n2 = True

    # outputs
    # n1Array, n2Array, timeArray, SS_check_n1, SS_check_n2
    return n1Array, n2Array, np.arange(N_time)*dt, SS_check_n1, SS_check_n2



# %%%%%%%%%%%%%%%%%%% M tilde: the form of the morphogen for a given cell (co-moving frame) %%%%%%%%%%%%%%%%%%%

def find_m_tilde_constant_growth(L0,g0,epsilon,sourceSize,vm,km,Dm,x_initial,timeArray):
    """ A function to find the analytical solution for m tilde (the non-dimensional morphogen concentration) at a given "cell" or co-moving coordinate (starting at x_initial) over time 
    for a tissue constantly growing at rate g0, with a fixed source size."""
    length_ev_term = np.exp(g0*timeArray/(1+epsilon))
    L_ana =  L0*length_ev_term
    x_ana = x_initial*length_ev_term
    lam = np.sqrt(Dm/km)
    source_amplitude = (vm/km)
    if x_initial < sourceSize:
        # starts in source 
        index_exit_source = np.searchsorted(x_ana,sourceSize)
        m_tilde = np.zeros(len(timeArray))
        m_tilde[:index_exit_source] = source_amplitude * (1 + np.sinh((sourceSize-L_ana[:index_exit_source])/lam)*np.cosh( x_ana[:index_exit_source] /lam)/np.sinh(L_ana[:index_exit_source]/lam)) 
        m_tilde[index_exit_source:] = source_amplitude * (np.sinh(sourceSize/lam) / np.sinh(L_ana[index_exit_source:]/lam)) * np.cosh((L_ana[index_exit_source:] - x_ana[index_exit_source:] )/lam) # outside the source
    else:
        m_tilde = source_amplitude * (np.sinh(sourceSize/lam) / np.sinh(L_ana/lam)) * np.cosh((L_ana - x_ana) /lam) # outside the source
    # m_tilde_i[np.isnan(m_tilde_i)] = 0 # sometimes the values get too small
    return m_tilde


def find_m_tilde_constant_growth_growing_source(L0,g0,epsilon,beta,gamma,vm,km,Dm,x_initial,timeArray):
    """ A function to find the analytical solution for m tilde (the non-dimensional morphogen concentration) at a given "cell" or co-moving coordinate (starting at x_initial) over time 
    for a tissue constantly growing at rate g0, with a scaling source size. 
        L0 = initial length of tissue
        g0 = (constant) growth rate
        epsilon = anisotropy parameter
        beta, gamma = parameters for source size scaling: source size = beta*L^gamma
        vm = morphogen production rate
        km = morphogen decay rate
        Dm = morphogen diffusion coefficient
        x_initial = initial position of the cell
    """
    length_ev_term = np.exp(g0*timeArray/(1+epsilon))
    L_ana =  L0*length_ev_term
    x_ana = x_initial*length_ev_term
    sourceSize = beta * L_ana**gamma # source size grows with the tissue size
    lam = np.sqrt(Dm/km)
    source_amplitude = (vm/km)
    m_tilde = np.zeros(len(timeArray)) # initiate m tilde
    # iterate over time & check for each x value if it is in the source
    for i in range(len(timeArray)):
        if x_ana[i] < sourceSize[i]: # if cell is within the source
            m_tilde[i] = source_amplitude * (1 + np.sinh((sourceSize[i]-L_ana[i])/lam)*np.cosh( x_ana[i] /lam)/np.sinh(L_ana[i]/lam)) 
        else:
            m_tilde[i] = source_amplitude * (np.sinh(sourceSize[i]/lam) / np.sinh(L_ana[i]/lam)) * np.cosh((L_ana[i] - x_ana[i]) /lam)

    return m_tilde

# %%%%%%%%%%%%%%%%%%% critical curve of bistability equations %%%%%%%%%%%%%%%%%%%

def pos_hill_function(m_value,h):
    return m_value**(h)/(1 + m_value**(h))

def find_c1(m_value,h,v1,k1):
    return (v1/k1)*(m_value**h/(1 + m_value**h))

def find_c2(v2,k2):
    return v2/k2

def critical_curve_eq(c1,c2,q,p,n2,n1):
    return 1 - c1*c2*q*p*(n2**q)*(n1**p) / ( ( ( 1 + n1**p ) *( 1 + n2**q ) )**2 ) # = 0 on the critical curve

def find_c1_parametric_critical_curve(n1,p,q):
    """ Function that returns c1 as a function of X along the critical curve """
    return p*q*(n1**(q+1) ) / ( (p*q - 1)* n1**q - 1 ) 

def find_c2_parametric_critical_curve(n1,p,q):
    """ Function that returns c2 as a function of X along the critical curve """
    return (1 + n1**q) * ( (1 + n1**q)/( (p*q - 1)*(n1**q) - 1 ) )**(1/p) 

def find_min_n1_bound_for_plotting_c1_c2_parametric(q,p):
    return 1/(q*p - 1)**(1/q)

def c2_parametric_critical_curve_equation_0(n1,p,q,c2): # function that equals 0
    """ Equation from rearranging c2 parametric critical curve to equal 0. Solution to this equation gives the critical values of X given p, q, and c2 """
    return (1 + n1**q) * ( (1 + n1**q)/( (p*q - 1)*(n1**q) - 1 ) )**(1/p) - c2

def c2_parametric_critical_curve_equation_0_simple(n1,p,q,c2): # simpler function that equals 0 - the above equation to the power of p
    """ Simpler equation that  """
    return ( (1 + n1**q)**p ) * ( (1 + n1**q)/( (p*q - 1)*(n1**q) - 1 ) ) - c2**p

def find_m_from_c1(c1,v1,k1,h):
    """ Given a value of c1, the production rate v1, the Hill parameters n1star and h, use the definition of c1 to find the M, morphogen concentration to give this value  """
    return (v1/(k1*c1) - 1) ** (-1/h)

def find_critical_m_values(v1, v2, k1, k2, q, p, h, estimate_ratio=1): 
    """ Function that finds the critical values of the non-dimensionalised morphogen m for given non-dim GRN parameterrs."""
    if q*p == 1:
        # no critical values exist
        return np.nan, np.nan, np.nan, np.nan
    c2_value = v2 / k2 # find c2, a function of the given constants
    X_split = ( (q + 1)/(q*p -1) )**((1/q))
    X_est_high = 100 # X is a normalised value. This is a high estimate to try and get the higher critical value
    min_X_bound = 1/(q*p - 1)**(1/q) # in order for c2 to be real, X must be larger than min_X_bound
    X_est_low = min_X_bound*estimate_ratio # this is to try and find the value just above the minimum X bound
    # old solver, using fsolve
    # X_critical_values = [fsolve(c2_parametric_critical_curve_equation_0_simple,X_est_high,args=(p,q,c2_value),maxfev=1000000),fsolve(c2_parametric_critical_curve_equation_0_simple,X_est_low,args=(p,q,c2_value),maxfev=1000000)]
    # solve separately for 2 values
    initial_X_ratio = 1.0000000001 
    initial_X_high = 1e10
    solution_1_converges = False
    solution_2_converges = False
    
    # sol 1 (large value of c1crit) try except loop
    try:
        sol1 = root_scalar(c2_parametric_critical_curve_equation_0_simple,args=(p,q,c2_value),bracket=[X_split,initial_X_high],x0=X_est_high)
        solution_1_converges = True 
    except ValueError: # if the solver fails with this estimate
        initial_X_high = 1e15
        try:
            sol1 = root_scalar(c2_parametric_critical_curve_equation_0_simple,args=(p,q,c2_value),bracket=[X_split,initial_X_high],x0=X_est_high)
            solution_1_converges = True 
        except ValueError:
            solution_1_converges = False
    
    # sol 2 (small value of c1crit) try except loop
    try:
        sol2 = root_scalar(c2_parametric_critical_curve_equation_0_simple,args=(p,q,c2_value),bracket=[min_X_bound*initial_X_ratio,X_split],x0=X_est_low)
        solution_2_converges = True 
    except ValueError:
        initial_X_ratio = 1.00000000000001
        try:
            sol2 = root_scalar(c2_parametric_critical_curve_equation_0_simple,args=(p,q,c2_value),bracket=[min_X_bound*initial_X_ratio,X_split],x0=X_est_low)
            solution_2_converges = True
        except ValueError:
            solution_2_converges = False

    
    # from solution roots (X), find the critical values of c1 for this value of c2
    if solution_1_converges == False:
        c1_critical1 = np.nan
    else:
        c1_critical1 = find_c1_parametric_critical_curve(sol1.root,p,q)
    
    if solution_2_converges == False:
        c1_critical2 = np.nan
    else:
        c1_critical2 = find_c1_parametric_critical_curve(sol2.root,p,q)


    
    # from c1 critical values find m 
    # c1 has a maximum value of alpha1, aka V1/(k1*n1star)
    if c1_critical1 > (v1/k1):
        # print("find_critical_m_values: Higher critical value (c1critical1) not reached for any m value")
        # in the case where there is no m_critical1, just find m_critical2
        m_critical2 = find_m_from_c1(c1_critical2,v1,k1,h)
        return c1_critical1, c1_critical2, np.nan, m_critical2
    
    if c1_critical2 > (v1/k1):
        # print("find_critical_m_values: Higher critical value (c1critical2) not reached for any m value")
        # in the case where there is no m_critical2, just find m_critical1
        m_critical1 = find_m_from_c1(c1_critical1,v1,k1,h)
        print("v1/k1 = ", v1/k1, "; c1_critical2 = ", c1_critical2, "not reached")
        return c1_critical1, c1_critical2, m_critical1, np.nan
    
    m_critical1 = find_m_from_c1(c1_critical1,v1,k1,h)
    m_critical2 = find_m_from_c1(c1_critical2,v1,k1,h)
    # print(type(m_critical1))
    # print(type(c1_critical1))

    if type(m_critical1) == np.ndarray:
        # print('m_critical1')
        m_critical1 = m_critical1[0]

    if type(m_critical2) == np.ndarray:
        # print('m_critical2')
        m_critical2 = m_critical2[0]

    if type(c1_critical1) == np.ndarray:
        # print('c1_critical1')
        c1_critical1 = c1_critical1[0]

    if type(c1_critical2) == np.ndarray:
        # print('c1_critical2')
        c1_critical2 = c1_critical2[0]

    return c1_critical1, c1_critical2, m_critical1, m_critical2

def find_plot_critical_curve(q,p,max_n1_bound=2,no_points=1000):
    """ Function that takes the Hill coefficients and a maximum value of the normalised n1 concentration and returns n1_values, c1_values, and c2_values
        This function uses the critical c1 and c2 functions above to make it easier to plot the critical values of c1 vs c2. 
        max_n1_bound 
        """
    min_n1_bound = find_min_n1_bound_for_plotting_c1_c2_parametric(q,p) 
    n1_values = np.logspace(np.log10(min_n1_bound),np.log10(max_n1_bound),no_points)[1:]
    c1_values = find_c1_parametric_critical_curve(n1_values,p,q)
    c2_values = find_c2_parametric_critical_curve(n1_values,p,q)
    return n1_values, c1_values, c2_values

def find_x_critical_from_m_critical(mCurrent,m_critical,xCurrent,N):
    """ Function that numerically finds the critical x position between stability given the critical normalised morphogen value and the current normalised morphogen profile.
        This is the same function as in the dimensional file. """
    backwards_conc = np.flip(mCurrent)
    backwards_x = np.flip(xCurrent)
    post_index = np.searchsorted(backwards_conc,m_critical)
    if post_index == N: # if C_thr > than even C[0]
        print("find_x_critical_from_m_critical: Mcritical is too high >M(x=0) - not reached for this morphogen gradient")
        x_critical = 0
    elif post_index == 0:
        print("find_x_critical_from_m_critical: Mcritical is too low  <M(x=N) - not reached for this morphogen gradient")
        x_critical = backwards_x[0]
    else:
        pre_index = post_index - 1
        cStep = backwards_conc[post_index] - backwards_conc[pre_index] # difference in concentration
        xStep = backwards_x[post_index]- backwards_x[pre_index] # difference in space
        c_Error = backwards_conc[post_index] - m_critical # the error in concentration from the index before the threshold, the overshoot
        x_Error = xStep*c_Error/cStep
        x_critical = backwards_x[post_index] - x_Error
    if type(x_critical) == np.ndarray:
        x_critical = x_critical[0]
    return x_critical

def find_bifurcation_diagram(
        m_critical1, m_critical2, p, q, h, v1, v2, k1, k2,
        lowest_m = 1e-2, highest_m = 1e2,
        estimate_ratio=1.01,num_m_val=100,number_of_points = 500,
        starting_estimate_higher = [1,-2],
        starting_estimate_unstable = [0,0], 
        starting_estimate_lower = [-4,2.5],
        n1min= 10**-2,n1max = 10**2,n2min =10**-2, n2max = 10**2,
        starting_estimate_higher_inputted = True
    ):
    """ Function that finds the bifurcation diagram for given non-dimensional parameters."""
    
    if np.isnan(m_critical2): # if no bifurcation occurs for high m
        m_critical2 = highest_m

    # if np.isnan(m_critical1): # does this ever happen?
    # use the critical values to generate M values for solving  
    unstable_m_values = np.logspace(np.log10(m_critical1*estimate_ratio), np.log10(m_critical2/estimate_ratio),num_m_val)
    upper_m_values = np.logspace(np.log10(m_critical1),np.log10(highest_m),num_m_val)
    lower_m_values = np.logspace(np.log10(lowest_m),np.log10(m_critical2),num_m_val)

    # initiate arrays for the n1 and n2 solutions
    unstable_n1_solArray = np.array([])
    upper_n1_solArray = np.array([])
    lower_n1_solArray = np.array([])
    unstable_n2_solArray = np.array([])
    upper_n2_solArray = np.array([])
    lower_n2_solArray = np.array([])


    for i in range(len(unstable_m_values)):
        m_val = unstable_m_values[i]
        
        # Interpolate both nullclines
        n1_values = np.logspace(np.log10(n1min), np.log10(n1max), number_of_points)
        n2_values = np.logspace(np.log10(n2min), np.log10(n2max), number_of_points)
        n1_SS_values = find_n1_SS_given_n2(n2_values,v1,k1,m_val,h,p)
        n2_SS_values = find_n2_SS_given_n1(n1_values,v2,k2,q) 


        f_n1_of_n2 = interp1d(np.log10(n2_values), np.log10(n1_SS_values), bounds_error=False, fill_value=np.nan)
        f_n2_of_n1 = interp1d(np.log10(n1_values), np.log10(n2_SS_values), bounds_error=False, fill_value=np.nan)
        # Define the function to find where the difference between the two is zero
        def difference_in_nullclines(logN):
            logn1, logn2 = logN
            val1 = f_n1_of_n2(logn2) - logn1
            val2 = f_n2_of_n1(logn1) - logn2
            return [val1, val2]
        
        unstable_logn1_sol, unstable_logn2_sol = fsolve(difference_in_nullclines, starting_estimate_unstable)
        unstable_n1_sol = 10**unstable_logn1_sol
        unstable_n2_sol = 10**unstable_logn2_sol 
        unstable_n1_solArray = np.append(unstable_n1_solArray,unstable_n1_sol)
        unstable_n2_solArray = np.append(unstable_n2_solArray,unstable_n2_sol)

    for i in range(len(upper_m_values)):
        m_val = upper_m_values[i]
        number_of_points = 500
        # high n1 solution
        n1_values = np.logspace(np.log10(n1min), np.log10(n1max), number_of_points)
        n2_values = np.logspace(np.log10(n2min), np.log10(n2max), number_of_points)
        n1_SS_values = find_n1_SS_given_n2(n2_values,v1,k1,m_val,h,p)
        n2_SS_values = find_n2_SS_given_n1(n1_values,v2,k2,q) 

        f_n1_of_n2 = interp1d(np.log10(n2_values), np.log10(n1_SS_values), bounds_error=False, fill_value=np.nan)
        f_n2_of_n1 = interp1d(np.log10(n1_values), np.log10(n2_SS_values), bounds_error=False, fill_value=np.nan)
        def difference_in_nullclines(logN):
            logn1, logn2 = logN
            val1 = f_n1_of_n2(logn2) - logn1
            val2 = f_n2_of_n1(logn1) - logn2
            return [val1, val2]
        
        if starting_estimate_higher_inputted == False:
            n1_amp = pos_hill_function(m_val,h) * v1/k1
            starting_estimate_higher = [np.log10(n1_amp),np.log10(find_n2_SS_given_n1(n1_amp,v2,k2,q))]

        high_logn1_sol, high_logn2_sol = fsolve(difference_in_nullclines, starting_estimate_higher)
        high_n1_sol = 10**high_logn1_sol
        high_n2_sol = 10**high_logn2_sol 
        upper_n1_solArray = np.append(upper_n1_solArray,high_n1_sol)
        upper_n2_solArray = np.append(upper_n2_solArray,high_n2_sol)

    for i in range(len(lower_m_values)):
        m_val = lower_m_values[i]
        number_of_points = number_of_points
        # low n1 solution
        n1_values = np.logspace(np.log10(n1min), np.log10(n1max), number_of_points)
        n2_values = np.logspace(np.log10(n2min), np.log10(n2max), number_of_points)
        n1_SS_values = find_n1_SS_given_n2(n2_values,v1,k1,m_val,h,p)
        n2_SS_values = find_n2_SS_given_n1(n1_values,v2,k2,q) 

        f_n1_of_n2 = interp1d(np.log10(n2_values), np.log10(n1_SS_values), bounds_error=False, fill_value=np.nan)
        f_n2_of_n1 = interp1d(np.log10(n1_values), np.log10(n2_SS_values), bounds_error=False, fill_value=np.nan)
        def difference_in_nullclines(logN):
            logn1, logn2 = logN
            val1 = f_n1_of_n2(logn2) - logn1
            val2 = f_n2_of_n1(logn1) - logn2
            return [val1, val2]
        

        n1_starting_estimate = find_n1_SS_given_n2(v2/k2,v1,k1,m_val,h,p)
        starting_estimate_lower = [np.log10(n1_starting_estimate),np.log10(v2/k2)]
        low_logn1_sol, low_logn2_sol = fsolve(difference_in_nullclines, starting_estimate_lower)
        low_n1_sol = 10**low_logn1_sol
        low_n2_sol = 10**low_logn2_sol 
        lower_n1_solArray = np.append(lower_n1_solArray,low_n1_sol)
        lower_n2_solArray = np.append(lower_n2_solArray,low_n2_sol)
    

    return unstable_m_values, unstable_n1_solArray, unstable_n2_solArray, upper_m_values, upper_n1_solArray, upper_n2_solArray, lower_m_values, lower_n1_solArray, lower_n2_solArray


# %%
