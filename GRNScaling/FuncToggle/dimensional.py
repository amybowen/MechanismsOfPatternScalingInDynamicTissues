# A set of function relating to the two node, mutual inhibitory toggle switch induced by a morphogen
# In general, the node activated by the morphogen is referred to as N1, and the other node as N2.
# The combination of activation by morphogen and inhibition on N1 is multiplicative Hill functions defined by a Hill coefficient and a critical concentration.

import numpy as np
import numba
import math
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.optimize import root_scalar

from FuncTogglev2.nondimensional4D import c2_parametric_critical_curve_equation_0_simple, find_c1_parametric_critical_curve, find_c2_parametric_critical_curve, find_min_n1_bound_for_plotting_c1_c2_parametric
# to sort
# import FuncAnalytical # to sort

""" Variables:
# N1 is the concentration of gene product activated by the morphogen
# N2 is the concentration of the other gene product
# n1 is the normalised concentration of N1, X=N1/N1star
# n2 is the normalised concentration of N2, Y=N2/N2star

# Parameters
# N1star is the critical Hill concentration of N1 for inhibition of N2
# N2star is the critical Hill concentration of N2 for inhibition of N1
# Mstar is the critical Hill concentration of morphogen M for activation of N1
# p is the Hill coefficient for inhibition of N1 by N2
# q is the Hill coefficient for inhibition of N2 by N1
# h is the Hill coefficient for activation of N1 by M

# Composite parameters:
# c1 = ( V1 / ( k1 * N1star ) ) * ( M / Mstar )^h / ( 1 + ( M / Mstar )^h )  
# c2 = V2 / ( k2 * N2star ) """



# %%%%%%%%%%%%%%%%%%% update equations for N1 and N2 %%%%%%%%%%%%%%%%%%%
def update_conc_N1_NoGrowth(MPrev,N1Prev,N2Prev,V1,k1,Mstar,N2star,h,p,dt):
    return N1Prev*(1-k1*dt) +  V1*dt*(1/( (Mstar/MPrev)**h + 1)) * (1 / (1 + (N2Prev/N2star)**p ) )

def update_conc_N2_NoGrowth(N1Prev,N2Prev,V2,k2,q,N1star,dt):
    return N2Prev*(1-k2*dt) + V2*dt / (1 + (N1Prev/N1star)**q )

# ODEs: update concentrations given concentrations at previous timesteps - with growth as additional dilution
def update_conc_N1_Growth(MPrev,N1Prev,N2Prev,gCurrent,V1,k1,Mstar,N2star,p,h,dt):
    return N1Prev*(1-k1*dt) +  V1*dt*(1/( (Mstar/MPrev)**h + 1)) * (1 / (1 + (N2Prev/N2star)**p ) ) - gCurrent*N1Prev

def update_conc_N2_Growth(N1Prev,N2Prev,gCurrent,V2,k2,q,N1star,dt):
    return N2Prev*(1-k2*dt) + V2*dt / (1 + (N1Prev/N1star)**q ) - gCurrent*N2Prev


# solvers for M 

@numba.njit
def M_solver_NoGrowth(N,N_1,N_2,w,D,k,VM,dt,dx,store_no,M_init:np.ndarray,count_max):
    # set initial conditions
    MCurrent = np.zeros(N)
    MPrev  = np.zeros(N)
    MCurrent[:] = M_init[:]

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)
    MArray =  np.zeros((no_of_stores,N))

    # set the values to the initial values
    MArray[0,:] =  M_init[:]

    for count in range(1,count_max):
        MPrev[:] = MCurrent[:]

        # for loop to update the morphogen, N1 and N2
        # within source
        for i in range(1,w): # iterate across space
            MCurrent[i] = MPrev[i]*(1-k*dt) + VM*dt + (D*dt/(dx**2))*(MPrev[i-1] + MPrev[i+1] -2*MPrev[i] )
        # outside of source
        for i in range(w,N_1):
            MCurrent[i] = MPrev[i]*(1-k*dt) + (D*dt/(dx**2))*(MPrev[i-1] + MPrev[i+1] -2*MPrev[i] )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        MCurrent[0] = MPrev[0]*(1-k*dt) + VM*dt + (D*dt/(dx**2))*(MPrev[1] - MPrev[0] )
        MCurrent[N_1] = MPrev[N_1]*(1-k*dt) + (D*dt/(dx**2))*(MPrev[N_2] - MPrev[N_1] )

        # store solutions every store_no
        if count % store_no == 0:
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            store_index = int(count/store_no)
            MArray[store_index,:] = MCurrent[:]


    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, MArray

# %%%%%%%%%%%%%%%%%%% 1D solvers for N1 and N2 %%%%%%%%%%%%%%%%%%%
@numba.njit
def N1N2M_solver_NoGrowth(N,N_1,N_2,w,D,k,VM,dt,dx,V1,h,p,k1,V2,q,k2,Mstar,N1star,N2star,store_no,M_init:np.ndarray,N1_init:np.ndarray,N2_init:np.ndarray,count_max):
    # set initial conditions
    MCurrent = np.zeros(N)
    N1Current = np.zeros(N)
    N2Current = np.zeros(N)

    MPrev  = np.zeros(N)
    N2Prev = np.zeros(N)
    N1Prev = np.zeros(N)

    MCurrent[:] = M_init[:]
    N1Current[:] = N1_init[:]
    N2Current[:] = N2_init[:]

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)

    MArray =  np.zeros((no_of_stores,N))
    N1Array = np.zeros((no_of_stores,N))
    N2Array = np.zeros((no_of_stores,N))

    # set the values to the initial values
    MArray[0,:] =  M_init[:]
    N1Array[0,:] = N1_init[:]
    N2Array[0,:] = N2_init[:]

    for count in range(1,count_max):
        MPrev[:] = MCurrent[:]
        N1Prev[:] = N1Current[:]
        N2Prev[:] = N2Current[:]

        # for loop to update the morphogen, N1 and N2
        # within source
        for i in range(1,w): # iterate across space
            MCurrent[i] = MPrev[i]*(1-k*dt) + VM*dt + (D*dt/(dx**2))*(MPrev[i-1] + MPrev[i+1] -2*MPrev[i] )
        # outside of source
        for i in range(w,N_1):
            MCurrent[i] = MPrev[i]*(1-k*dt) + (D*dt/(dx**2))*(MPrev[i-1] + MPrev[i+1] -2*MPrev[i] )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        MCurrent[0] = MPrev[0]*(1-k*dt) + VM*dt + (D*dt/(dx**2))*(MPrev[1] - MPrev[0] )
        MCurrent[N_1] = MPrev[N_1]*(1-k*dt) + (D*dt/(dx**2))*(MPrev[N_2] - MPrev[N_1] )

        for i in range(N): # iterate across space for gene products
            N1Current[i] = N1Prev[i]*(1-k1*dt) + V1*dt*(MCurrent[i]**h/(Mstar**h + MCurrent[i]**h)) * (1 / (1 + (N2Prev[i]/N2star)**p ) )
            N2Current[i] = N2Prev[i]*(1-k2*dt) + V2*dt / (1 + (N1Prev[i]/N1star)**q )


        # store solutions every store_no
        if count % store_no == 0:
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            store_index = int(count/store_no)
            MArray[store_index,:] = MCurrent[:]
            N1Array[store_index,:] = N1Current[:]
            N2Array[store_index,:] = N2Current[:]


    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, MArray, N1Array, N2Array

@numba.njit
def N1N2M_solver_NoGrowth_SS(N,N_1,N_2,w,D,k,VM,dt,dx,V1,h,p,k1,V2,q,k2,Mstar,N1star,N2star,store_no,M_init:np.ndarray,N1_init:np.ndarray,N2_init:np.ndarray,count_max,SS_threshold=1E-5,SS_proportion=0.9):
    # set initial conditions
    MCurrent = np.zeros(N)
    N1Current = np.zeros(N)
    N2Current = np.zeros(N)

    MPrev  = np.zeros(N)
    N2Prev = np.zeros(N)
    N1Prev = np.zeros(N)

    MCurrent[:] = M_init[:]
    N1Current[:] = N1_init[:]
    N2Current[:] = N2_init[:]

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)

    MArray =  np.zeros((no_of_stores,N))
    N1Array = np.zeros((no_of_stores,N))
    N2Array = np.zeros((no_of_stores,N))

    # set the values to the initial values
    MArray[0,:] =  M_init[:]
    N1Array[0,:] = N1_init[:]
    N2Array[0,:] = N2_init[:]

    # initial steady state flags
    flag_M = 0
    flag_n1 = 0
    flag_n2 = 0

    timeSS = [np.nan, np.nan, np.nan]
    for count in range(1,count_max):
        MPrev[:] = MCurrent[:]
        N1Prev[:] = N1Current[:]
        N2Prev[:] = N2Current[:]

        # for loop to update the morphogen, N1 and N2
        # within source
        for i in range(1,w): # iterate across space
            MCurrent[i] = MPrev[i]*(1-k*dt) + VM*dt + (D*dt/(dx**2))*(MPrev[i-1] + MPrev[i+1] -2*MPrev[i] )
        # outside of source
        for i in range(w,N_1):
            MCurrent[i] = MPrev[i]*(1-k*dt) + (D*dt/(dx**2))*(MPrev[i-1] + MPrev[i+1] -2*MPrev[i] )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        MCurrent[0] = MPrev[0]*(1-k*dt) + VM*dt + (D*dt/(dx**2))*(MPrev[1] - MPrev[0] )
        MCurrent[N_1] = MPrev[N_1]*(1-k*dt) + (D*dt/(dx**2))*(MPrev[N_2] - MPrev[N_1] )

        for i in range(N): # iterate across space for gene products
            N1Current[i] = N1Prev[i]*(1-k1*dt) + V1*dt*(MCurrent[i]**h/(Mstar**h + MCurrent[i]**h)) * (1 / (1 + (N2Prev[i]/N2star)**p ) )
            N2Current[i] = N2Prev[i]*(1-k2*dt) + V2*dt / (1 + (N1Prev[i]/N1star)**q )


        # store solutions every store_no
        if count % store_no == 0:
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            store_index = int(count/store_no)
            MArray[store_index,:] = MCurrent[:]
            N1Array[store_index,:] = N1Current[:]
            N2Array[store_index,:] = N2Current[:]
            # check steady state
            # morphogen steady state
            dM = (MCurrent - MPrev)/ (MCurrent*dt)
            M_flag_count = np.sum(dM < SS_threshold)
            if M_flag_count > SS_proportion*N and flag_M == 0:
                timeSS[0] = count*dt
                flag_M = 1
            # n1 steady state
            dn1 = (N1Current - N1Prev)/ (N1Current*dt)
            n1_flag_count = np.sum(dn1 < SS_threshold)
            if n1_flag_count > SS_proportion*N and flag_n1 == 0:
                timeSS[1] = count*dt
                flag_n1 = 1
            # n2 steady state
            dn2 = (N2Current - N2Prev)/ (N2Current*dt)
            n2_flag_count = np.sum(dn2 < SS_threshold)
            if n2_flag_count > SS_proportion*N and flag_n2 == 0:
                timeSS[2] = count*dt
                flag_n2 = 1
            
            if flag_M == 1 and flag_n1 == 1 and flag_n2 == 1:
                timeArray = np.linspace(0,(count)*dt,math.ceil((count+1)/store_no))
                return timeArray, MArray[0:len(timeArray),:], N1Array[0:len(timeArray),:], N2Array[0:len(timeArray),:], timeSS
            


    # print(flag_M,flag_n1,flag_n2)
    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, MArray, N1Array, N2Array, timeSS

@numba.njit
def N1N2M_solver_NoGrowth_fixedM(N,dt,V1,Mstar,N1star,N2star,p,h,k1,V2,q,k2,store_no,N1_init:np.ndarray,N2_init:np.ndarray,M_init:np.ndarray,count_max):
    """ A solver for N1 and N2 (mutual inhibitory toggle switch) with a fixed morphogen profile M over time, with no growth.
        Essentially a set of 0D solvers. """
    # initiate initial conditions
    MCurrent = np.zeros(N)
    N1Current = np.zeros(N)
    N2Current = np.zeros(N)

    # initiate previous arrays for N1 and N2
    N2Prev = np.zeros(N)
    N1Prev = np.zeros(N)
    # set the current Arrays as the initial conditions
    MCurrent[:] = M_init
    N1Current[:] = N1_init[:]
    N2Current[:] = N2_init[:]

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)

    MArray =  np.zeros((no_of_stores,N))
    N1Array = np.zeros((no_of_stores,N))
    N2Array = np.zeros((no_of_stores,N))

    # set the values to the initial values
    MArray[0,:] = M_init
    N1Array[0,:] = N1_init[:]
    N2Array[0,:] = N2_init[:]

    for count in range(1,count_max):
        # MPrev[:] = MCurrent[:]
        N1Prev[:] = N1Current[:]
        N2Prev[:] = N2Current[:]

        # for loop to update the morphogen, N1 and N2
        # within source
        # for i in range(1,w): # iterate across space
        #     MCurrent[i] = MPrev[i]*(1-k*dt) + VM*dt + (D*dt/(dx**2))*(MPrev[i-1] + MPrev[i+1] -2*MPrev[i] )
        # # outside of source
        # for i in range(w,N_1):
        #     MCurrent[i] = MPrev[i]*(1-k*dt) + (D*dt/(dx**2))*(MPrev[i-1] + MPrev[i+1] -2*MPrev[i] )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        # MCurrent[0] = MPrev[0]*(1-k*dt) + VM*dt + (D*dt/(dx**2))*(MPrev[1] - MPrev[0] )
        # MCurrent[N_1] = MPrev[N_1]*(1-k*dt) + (D*dt/(dx**2))*(MPrev[N_2] - MPrev[N_1] )

        for i in range(N): # iterate across space for gene products
            N1Current[i] = N1Prev[i]*(1-k1*dt) + V1*dt*(MCurrent[i]**h/(Mstar**h + MCurrent[i]**h)) * (1 / (1 + (N2Prev[i]/N2star)**p ) )
            N2Current[i] = N2Prev[i]*(1-k2*dt) + V2*dt / (1 + (N1Prev[i]/N1star)**q )

        
        # store solutions every store_no
        if count % store_no == 0:
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            store_index = int(count/store_no)
            MArray[store_index,:] = MCurrent[:]
            N1Array[store_index,:] = N1Current[:]
            N2Array[store_index,:] = N2Current[:]


    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, MArray, N1Array, N2Array

@numba.njit
def M_solver_Growth_scale_source(N,N_1,N_2,wCurrent,D,k,epsilon,beta,gamma,VM,dt,dy,L0,store_no,M_init:np.ndarray,count_max,g_max,alpha):
    # N1N2M_solver_Growth(N,N_1,N_2,wCurrent,D,k,epsilon,sRatio,VM,dt,dy,V1,h,p,k1,V2,q,k2,Mstar,N1star,N2star,L0,store_no,M_init,N1_init,N2_init,count_max,g_max,alpha)
    # a loop to solve for C, N1, N2

    # initialise Current and Prev arrays for M, x_Y
    MCurrent = np.zeros(N)
    MPrev  = np.zeros(N)
    x_YPrev = np.zeros(N)

    # set initial conditions as current arrays
    xCurrent = np.linspace(0,L0,N) #The co-moving coordinates: N spatial points, includes the point 0 and L0
    MCurrent[:] = M_init[:]
    x_YCurrent = np.ones(N) # initial x_Y
    gCurrent = np.zeros(N)
    prodRat = 0

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)
    # 2D
    MArray =  np.zeros((no_of_stores,N))
    xArray = np.zeros((no_of_stores,N))
    gArray = np.zeros((no_of_stores,N))
    x_YArray = np.zeros((no_of_stores,N))
    # 1D
    LArray = np.zeros(no_of_stores)
    wArray = np.zeros(no_of_stores)

    # set the values to the initial values
    MArray[0,:] =  M_init[:]
    xArray[0,:] = xCurrent[:]
    x_YArray[0,:] = x_YCurrent[:]
    LArray[0] = L0
    wArray[0] = wCurrent

    for count in range(1,count_max):
        MPrev[:] = MCurrent[:]
        x_YPrev[:] = x_YCurrent[:]

        # MCurrent = FuncUpdate.update_conc(N,N_1,N_2,MPrev,MCurrent,D,dt,dy,x_YPrev,k,VM,sCurrent,gCurrent)
        # for loop to update the morphogen
        # within source
        for i in range(1,wCurrent): # iterate across space
            MCurrent[i] = MPrev[i]*(1-(k+gCurrent[i])*dt) + VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # at in between point
        MCurrent[wCurrent] = MPrev[wCurrent]*(1-(k+gCurrent[wCurrent])*dt) + prodRat*VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[wCurrent]) * ( ( (MPrev[wCurrent+1] - MPrev[wCurrent])/(x_YPrev[wCurrent+1] + x_YPrev[wCurrent]) ) - ( (MPrev[wCurrent] - MPrev[wCurrent-1])/(x_YPrev[wCurrent-1] + x_YPrev[wCurrent]) ) ) )
        # outside of source

        for i in range(wCurrent+1,N_1):
            MCurrent[i] = MPrev[i]*(1-(k+gCurrent[i])*dt) + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        MCurrent[0] = MPrev[0]*(1-(k+gCurrent[i])*dt) + VM*dt + (D*dt/(dy**2))*(MPrev[1] - MPrev[0] )
        MCurrent[N_1] = MPrev[N_1]*(1-(k+gCurrent[i])*dt) + (D*dt/(dy**2))*(MPrev[N_2] - MPrev[N_1] )

        Sx_Y = 0 # initiate Sum(x_Y) over Y, the value of x_Y at x=0 (Y=0) for all time
        for i in range(N): # iterate across space for gene products
            gCurrent[i] = g_max*np.exp(-(count*dt)*alpha)
            x_YCurrent[i] = x_YPrev[i]/(1-gCurrent[i]*dt/(1+epsilon))
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
        

        # # output solutions every store_no
        if count % store_no == 0:
            store_index = int(count/store_no)
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            MArray[store_index,:] = MCurrent[:]
            xArray[store_index,:] = xCurrent[:]
            gArray[store_index,:] = gCurrent[:]
            x_YArray[store_index,:] = x_YCurrent[:]
            LArray[store_index] = LCurrent
            wArray[store_index] = wCurrent

    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, xArray, MArray, gArray, x_YArray, LArray, wArray


@numba.njit
def N1N2M_solver_Growth_scale_source(N,N_1,N_2,wCurrent,D,k,epsilon,beta,gamma,VM,dt,dy,V1,h,p,k1,V2,q,k2,Mstar,N1star,N2star,L0,store_no,M_init:np.ndarray,N1_init:np.ndarray,N2_init:np.ndarray,count_max,g_max,alpha):
    # N1N2M_solver_Growth(N,N_1,N_2,wCurrent,D,k,epsilon,sRatio,VM,dt,dy,V1,h,p,k1,V2,q,k2,Mstar,N1star,N2star,L0,store_no,M_init,N1_init,N2_init,count_max,g_max,alpha)
    # a loop to solve for C, N1, N2

    # initialise Current and Prev arrays for M, N1, N2, x_Y
    MCurrent = np.zeros(N)
    N1Current = np.zeros(N)
    N2Current = np.zeros(N)

    MPrev  = np.zeros(N)
    N2Prev = np.zeros(N)
    N1Prev = np.zeros(N)
    x_YPrev = np.zeros(N)

    # set initial conditions as current arrays
    xCurrent = np.linspace(0,L0,N) #The co-moving coordinates: N spatial points, includes the point 0 and L0
    MCurrent[:] = M_init[:]
    N1Current[:] = N1_init[:]
    N2Current[:] = N2_init[:]
    x_YCurrent = np.ones(N) # initial x_Y
    gCurrent = np.zeros(N)
    prodRat = 0

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)
    # 2D
    MArray =  np.zeros((no_of_stores,N))
    N1Array = np.zeros((no_of_stores,N))
    N2Array = np.zeros((no_of_stores,N))
    xArray = np.zeros((no_of_stores,N))
    gArray = np.zeros((no_of_stores,N))
    x_YArray = np.zeros((no_of_stores,N))
    # 1D
    LArray = np.zeros(no_of_stores)
    wArray = np.zeros(no_of_stores)

    # set the values to the initial values
    MArray[0,:] =  M_init[:]
    N1Array[0,:] = N1_init[:]
    N2Array[0,:] = N2_init[:]
    xArray[0,:] = xCurrent[:]
    x_YArray[0,:] = x_YCurrent[:]
    LArray[0] = L0
    wArray[0] = wCurrent

    for count in range(1,count_max):
        MPrev[:] = MCurrent[:]
        N1Prev[:] = N1Current[:]
        N2Prev[:] = N2Current[:]
        x_YPrev[:] = x_YCurrent[:]

        # MCurrent = FuncUpdate.update_conc(N,N_1,N_2,MPrev,MCurrent,D,dt,dy,x_YPrev,k,VM,sCurrent,gCurrent)
        # for loop to update the morphogen
        # within source
        for i in range(1,wCurrent): # iterate across space
            MCurrent[i] = MPrev[i]*(1-(k+gCurrent[i])*dt) + VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # at in between point
        MCurrent[wCurrent] = MPrev[wCurrent]*(1-(k+gCurrent[wCurrent])*dt) + prodRat*VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[wCurrent]) * ( ( (MPrev[wCurrent+1] - MPrev[wCurrent])/(x_YPrev[wCurrent+1] + x_YPrev[wCurrent]) ) - ( (MPrev[wCurrent] - MPrev[wCurrent-1])/(x_YPrev[wCurrent-1] + x_YPrev[wCurrent]) ) ) )
        # outside of source

        for i in range(wCurrent+1,N_1):
            MCurrent[i] = MPrev[i]*(1-(k+gCurrent[i])*dt) + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        MCurrent[0] = MPrev[0]*(1-(k+gCurrent[i])*dt) + VM*dt + (D*dt/(dy**2))*(MPrev[1] - MPrev[0] )
        MCurrent[N_1] = MPrev[N_1]*(1-(k+gCurrent[i])*dt) + (D*dt/(dy**2))*(MPrev[N_2] - MPrev[N_1] )

        Sx_Y = 0 # initiate Sum(x_Y) over Y, the value of x_Y at x=0 (Y=0) for all time
        for i in range(N): # iterate across space for gene products
            N1Current[i] = N1Prev[i]*(1-k1*dt) + V1*dt*(MCurrent[i]**h/(Mstar**h + MCurrent[i]**h)) * (1 / (1 + (N2Prev[i]/N2star)**p ) ) - dt*gCurrent[i]*N1Prev[i]
            N2Current[i] = N2Prev[i]*(1-k2*dt) + V2*dt / (1 + (N1Prev[i]/N1star)**q ) - dt*gCurrent[i]*N2Prev[i]
            gCurrent[i] = g_max*np.exp(-(count*dt)*alpha)
            x_YCurrent[i] = x_YPrev[i]/(1-gCurrent[i]*dt/(1+epsilon))
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
        

        # # output solutions every store_no
        if count % store_no == 0:
            store_index = int(count/store_no)
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            MArray[store_index,:] = MCurrent[:]
            N1Array[store_index,:] = N1Current[:]
            N2Array[store_index,:] = N2Current[:]
            xArray[store_index,:] = xCurrent[:]
            gArray[store_index,:] = gCurrent[:]
            x_YArray[store_index,:] = x_YCurrent[:]
            LArray[store_index] = LCurrent
            wArray[store_index] = wCurrent

    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, xArray, MArray, N1Array, N2Array, gArray, x_YArray, LArray, wArray


# growing solver that doesn't scale the source
@numba.njit
def N1N2M_solver_Decaying_Growth_fixed_source(N,N_1,N_2,w0,D,k,epsilon,sourceSize,VM,dt,dy,V1,h,p,k1,V2,q,k2,Mstar,N1star,N2star,L0,store_no,M_init:np.ndarray,N1_init:np.ndarray,N2_init:np.ndarray,count_max,g_max,alpha):    
    """ A solver for N1 and N2 (mutual inhibitory toggle switch) with a fixed morphogen profile M over time with growth.
        Essentially a set of 0D solvers. """
    # a loop to solve for M, N1, N2

    # initialise Current and Prev arrays for M, N1, N2, x_Y
    MCurrent = np.zeros(N)
    N1Current = np.zeros(N)
    N2Current = np.zeros(N)

    MPrev  = np.zeros(N)
    N2Prev = np.zeros(N)
    N1Prev = np.zeros(N)
    x_YPrev = np.zeros(N)

    # set initial conditions as current arrays
    xCurrent = np.linspace(0,L0,N) #The co-moving coordinates: N spatial points, includes the point 0 and L0
    MCurrent[:] = M_init[:]
    N1Current[:] = N1_init[:]
    N2Current[:] = N2_init[:]
    x_YCurrent = np.ones(N) # initial x_Y
    gCurrent = np.zeros(N)
    wCurrent = w0
    prodRat = 0

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)
    # 2D
    MArray =  np.zeros((no_of_stores,N))
    N1Array = np.zeros((no_of_stores,N))
    N2Array = np.zeros((no_of_stores,N))
    xArray = np.zeros((no_of_stores,N))
    gArray = np.zeros((no_of_stores,N))
    x_YArray = np.zeros((no_of_stores,N))
    # 1D
    LArray = np.zeros(no_of_stores)
    wArray = np.zeros(no_of_stores)

    # set the values to the initial values
    MArray[0,:] =  M_init[:]
    N1Array[0,:] = N1_init[:]
    N2Array[0,:] = N2_init[:]
    xArray[0,:] = xCurrent[:]
    x_YArray[0,:] = x_YCurrent[:]
    LArray[0] = L0
    wArray[0] = wCurrent

    for count in range(1,count_max):
        MPrev[:] = MCurrent[:]
        N1Prev[:] = N1Current[:]
        N2Prev[:] = N2Current[:]
        x_YPrev[:] = x_YCurrent[:]

        # MCurrent = FuncUpdate.update_conc(N,N_1,N_2,MPrev,MCurrent,D,dt,dy,x_YPrev,k,VM,sCurrent,gCurrent)

        # for loop to update the morphogen
        # within source
        for i in range(1,wCurrent): # iterate across space to the point before wCurrent
            MCurrent[i] = MPrev[i]*(1-(k+gCurrent[i])*dt) + VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # at the index wCurrent, which has a production prodRat
        MCurrent[wCurrent] = MPrev[wCurrent]*(1-(k+gCurrent[wCurrent])*dt) + prodRat*VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[wCurrent]) * ( ( (MPrev[wCurrent+1] - MPrev[wCurrent])/(x_YPrev[wCurrent+1] + x_YPrev[wCurrent]) ) - ( (MPrev[wCurrent] - MPrev[wCurrent-1])/(x_YPrev[wCurrent-1] + x_YPrev[wCurrent]) ) ) )
        # outside of source

        for i in range(wCurrent+1,N_1):
            MCurrent[i] = MPrev[i]*(1-(k+gCurrent[i])*dt) + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        MCurrent[0] = MPrev[0]*(1-(k+gCurrent[i])*dt) + VM*dt + (D*dt/(dy**2))*(MPrev[1] - MPrev[0] )
        MCurrent[N_1] = MPrev[N_1]*(1-(k+gCurrent[i])*dt) + (D*dt/(dy**2))*(MPrev[N_2] - MPrev[N_1] )

        # update x_Y and xArray
        Sx_Y = 0 # initiate Sum(x_Y) over Y, the value of x_Y at x=0 (Y=0) for all time
        for i in range(N): # iterate across space for gene products
            N1Current[i] = N1Prev[i]*(1-k1*dt) + V1*dt*(MCurrent[i]**h/(Mstar**h + MCurrent[i]**h)) * (1 / (1 + (N2Prev[i]/N2star)**p ) ) - dt*gCurrent[i]*N1Prev[i]
            N2Current[i] = N2Prev[i]*(1-k2*dt) + V2*dt / (1 + (N1Prev[i]/N1star)**q ) - dt*gCurrent[i]*N2Prev[i]
            gCurrent[i] = g_max*np.exp(-(count*dt)*alpha)
            x_YCurrent[i] = x_YPrev[i]/(1-gCurrent[i]*dt/(1+epsilon))
            xCurrent[i] = dy*Sx_Y
            Sx_Y += x_YCurrent[i]
    

        # update length of tissue L
        LCurrent = xCurrent[N_1]
      
        # update source width and corresponding index in co-moving coordinates
        # [wCurrent,sCurrent] = FuncUpdate.update_source(sRatio,N,LCurrent,xCurrent,sCurrent)
        wCurrent = np.searchsorted(xCurrent,sourceSize) # w is the index that sSizeIdeal should be placed in xCurrent to be in order. First index larger than sourcesize
        sStep = xCurrent[wCurrent] - xCurrent[wCurrent-1]
        sSizeError = sourceSize - xCurrent[wCurrent-1] # find the error in source size due to discretisation
        prodRat = sSizeError/sStep # the normalised fractional error in the source position
        

        # # output solutions every store_no
        if count % store_no == 0:
            store_index = int(count/store_no)
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            MArray[store_index,:] = MCurrent[:]
            N1Array[store_index,:] = N1Current[:]
            N2Array[store_index,:] = N2Current[:]
            xArray[store_index,:] = xCurrent[:]
            gArray[store_index,:] = gCurrent[:]
            x_YArray[store_index,:] = x_YCurrent[:]
            LArray[store_index] = LCurrent
            wArray[store_index] = wCurrent

    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, xArray, MArray, N1Array, N2Array, gArray, x_YArray, LArray, wArray

@numba.njit
def Constant_Growth_solver(N,N_1,epsilon,dt,dy,L0,store_no,count_max,g0):
    """ A solver for the length of tissue given a growth """
    # initialise Prev array for x_Y
    x_YPrev = np.zeros(N)
    # set initial conditions as current arrays
    
    xCurrent = np.linspace(0,L0,N) #The co-moving coordinates: N spatial points, includes the point 0 and L0
    x_YCurrent = np.ones(N) # initial x_Y

     # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)
    # 2D
    xArray = np.zeros((no_of_stores,N))
    x_YArray = np.zeros((no_of_stores,N))
    # 1D
    LArray = np.zeros(no_of_stores)

    # set the values to the initial values
    xArray[0,:] = xCurrent[:]
    x_YArray[0,:] = x_YCurrent[:]
    LArray[0] = L0

    for count in range(1,count_max):
        x_YPrev[:] = x_YCurrent[:]
        Sx_Y = 0 # initiate Sum(x_Y) over Y, the value of x_Y at x=0 (Y=0) for all time
        for i in range(N): # iterate across space for gene products
            x_YCurrent[i] = x_YPrev[i]/(1-g0*dt/(1+epsilon))
            xCurrent[i] = dy*Sx_Y
            Sx_Y += x_YCurrent[i]
    
        # update length of tissue L
        LCurrent = xCurrent[N_1]

        # output solutions every store_no
        if count % store_no == 0:
            store_index = int(count/store_no)
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            xArray[store_index,:] = xCurrent[:]
            x_YArray[store_index,:] = x_YCurrent[:]
            LArray[store_index] = LCurrent

    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, xArray, x_YArray, LArray

@numba.njit
def M_solver_ConstantGrowth_scale_source(N,N_1,N_2,wCurrent,D,k,epsilon,beta,gamma,VM,dt,dy,L0,store_no,M_init:np.ndarray,count_max,g0):  
    # a loop to solve for C, N1, N2

    # initialise Current and Prev arrays for M, x_Y
    MCurrent = np.zeros(N)

    MPrev  = np.zeros(N)
    x_YPrev = np.zeros(N)

    # set initial conditions as current arrays
    xCurrent = np.linspace(0,L0,N) #The co-moving coordinates: N spatial points, includes the point 0 and L0
    MCurrent[:] = M_init[:]
    x_YCurrent = np.ones(N) # initial x_Y
    prodRat = 0

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)
    # 2D
    MArray =  np.zeros((no_of_stores,N))
    xArray = np.zeros((no_of_stores,N))
    x_YArray = np.zeros((no_of_stores,N))
    # 1D
    LArray = np.zeros(no_of_stores)
    wArray = np.zeros(no_of_stores)

    # set the values to the initial values
    MArray[0,:] =  M_init[:]
    xArray[0,:] = xCurrent[:]
    x_YArray[0,:] = x_YCurrent[:]
    LArray[0] = L0
    wArray[0] = wCurrent

    for count in range(1,count_max):
        MPrev[:] = MCurrent[:]
        x_YPrev[:] = x_YCurrent[:]

        # for loop to update the morphogen
        # # within source
        for i in range(1,wCurrent): # iterate across space
            MCurrent[i] = MPrev[i]*(1-(k+g0)*dt) + VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # at in between point - taking into account the production ratio
        MCurrent[wCurrent] = MPrev[wCurrent]*(1-(k+g0)*dt) + prodRat*VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[wCurrent]) * ( ( (MPrev[wCurrent+1] - MPrev[wCurrent])/(x_YPrev[wCurrent+1] + x_YPrev[wCurrent]) ) - ( (MPrev[wCurrent] - MPrev[wCurrent-1])/(x_YPrev[wCurrent-1] + x_YPrev[wCurrent]) ) ) )

        # outside of source
        for i in range(wCurrent+1,N_1):
            MCurrent[i] = MPrev[i]*(1-(k+g0)*dt) + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        MCurrent[0] = MPrev[0]*(1-(k+g0)*dt) + VM*dt + (D*dt/(dy**2))*(MPrev[1] - MPrev[0] )
        MCurrent[N_1] = MPrev[N_1]*(1-(k+g0)*dt) + (D*dt/(dy**2))*(MPrev[N_2] - MPrev[N_1] )

        Sx_Y = 0 # initiate Sum(x_Y) over Y, the value of x_Y at x=0 (Y=0) for all time
        for i in range(N): # iterate across space for gene products
            x_YCurrent[i] = x_YPrev[i]/(1-g0*dt/(1+epsilon))
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
        

        # # output solutions every store_no
        if count % store_no == 0:
            store_index = int(count/store_no)
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            MArray[store_index,:] = MCurrent[:]
            xArray[store_index,:] = xCurrent[:]
            x_YArray[store_index,:] = x_YCurrent[:]
            LArray[store_index] = LCurrent
            wArray[store_index] = wCurrent

    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, xArray, MArray, x_YArray, LArray, wArray

@numba.njit
def N1N2M_solver_ConstantGrowth_scale_source(N,N_1,N_2,wCurrent,D,k,epsilon,beta,gamma,VM,dt,dy,V1,h,p,k1,V2,q,k2,Mstar,N1star,N2star,L0,store_no,M_init:np.ndarray,N1_init:np.ndarray,N2_init:np.ndarray,count_max,g0):  
    # a loop to solve for C, N1, N2

    # initialise Current and Prev arrays for M, N1, N2, x_Y
    MCurrent = np.zeros(N)
    N1Current = np.zeros(N)
    N2Current = np.zeros(N)

    MPrev  = np.zeros(N)
    N2Prev = np.zeros(N)
    N1Prev = np.zeros(N)
    x_YPrev = np.zeros(N)

    # set initial conditions as current arrays
    xCurrent = np.linspace(0,L0,N) #The co-moving coordinates: N spatial points, includes the point 0 and L0
    MCurrent[:] = M_init[:]
    N1Current[:] = N1_init[:]
    N2Current[:] = N2_init[:]
    x_YCurrent = np.ones(N) # initial x_Y
    prodRat = 0

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)
    # 2D
    MArray =  np.zeros((no_of_stores,N))
    N1Array = np.zeros((no_of_stores,N))
    N2Array = np.zeros((no_of_stores,N))
    xArray = np.zeros((no_of_stores,N))
    x_YArray = np.zeros((no_of_stores,N))
    # 1D
    LArray = np.zeros(no_of_stores)
    wArray = np.zeros(no_of_stores)

    # set the values to the initial values
    MArray[0,:] =  M_init[:]
    N1Array[0,:] = N1_init[:]
    N2Array[0,:] = N2_init[:]
    xArray[0,:] = xCurrent[:]
    x_YArray[0,:] = x_YCurrent[:]
    LArray[0] = L0
    wArray[0] = wCurrent

    for count in range(1,count_max):
        MPrev[:] = MCurrent[:]
        N1Prev[:] = N1Current[:]
        N2Prev[:] = N2Current[:]
        x_YPrev[:] = x_YCurrent[:]

        # for loop to update the morphogen
        # # within source
        for i in range(1,wCurrent): # iterate across space
            MCurrent[i] = MPrev[i]*(1-(k+g0)*dt) + VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # at in between point - taking into account the production ratio
        MCurrent[wCurrent] = MPrev[wCurrent]*(1-(k+g0)*dt) + prodRat*VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[wCurrent]) * ( ( (MPrev[wCurrent+1] - MPrev[wCurrent])/(x_YPrev[wCurrent+1] + x_YPrev[wCurrent]) ) - ( (MPrev[wCurrent] - MPrev[wCurrent-1])/(x_YPrev[wCurrent-1] + x_YPrev[wCurrent]) ) ) )

        # outside of source
        for i in range(wCurrent+1,N_1):
            MCurrent[i] = MPrev[i]*(1-(k+g0)*dt) + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        MCurrent[0] = MPrev[0]*(1-(k+g0)*dt) + VM*dt + (D*dt/(dy**2))*(MPrev[1] - MPrev[0] )
        MCurrent[N_1] = MPrev[N_1]*(1-(k+g0)*dt) + (D*dt/(dy**2))*(MPrev[N_2] - MPrev[N_1] )

        Sx_Y = 0 # initiate Sum(x_Y) over Y, the value of x_Y at x=0 (Y=0) for all time
        for i in range(N): # iterate across space for gene products
            N1Current[i] = N1Prev[i]*(1-k1*dt) + V1*dt*(MCurrent[i]**h/(Mstar**h + MCurrent[i]**h)) * (1 / (1 + (N2Prev[i]/N2star)**p ) ) - dt*g0*N1Prev[i]
            N2Current[i] = N2Prev[i]*(1-k2*dt) + V2*dt / (1 + (N1Prev[i]/N1star)**q ) - dt*g0*N2Prev[i]
            x_YCurrent[i] = x_YPrev[i]/(1-g0*dt/(1+epsilon))
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
        

        # # output solutions every store_no
        if count % store_no == 0:
            store_index = int(count/store_no)
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            MArray[store_index,:] = MCurrent[:]
            N1Array[store_index,:] = N1Current[:]
            N2Array[store_index,:] = N2Current[:]
            xArray[store_index,:] = xCurrent[:]
            x_YArray[store_index,:] = x_YCurrent[:]
            LArray[store_index] = LCurrent
            wArray[store_index] = wCurrent

    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, xArray, MArray, N1Array, N2Array, x_YArray, LArray, wArray

@numba.njit
def N1N2M_solver_ConstantGrowth_fixed_source(N,N_1,N_2,w0,D,k,epsilon,sourceSize,VM,dt,dy,V1,h,p,k1,V2,q,k2,Mstar,N1star,N2star,L0,store_no,M_init:np.ndarray,N1_init:np.ndarray,N2_init:np.ndarray,count_max,g0):
    """ A solver for M, N1 and N2 over time with constant growth. The morphogen source size is fixed. """ 

    # initialise Current and Prev arrays for M, N1, N2, x_Y
    MCurrent = np.zeros(N)
    N1Current = np.zeros(N)
    N2Current = np.zeros(N)

    MPrev  = np.zeros(N)
    N2Prev = np.zeros(N)
    N1Prev = np.zeros(N)
    x_YPrev = np.zeros(N)

    # set initial conditions as current arrays
    xCurrent = np.linspace(0,L0,N) #The co-moving coordinates: N spatial points, includes the point 0 and L0
    MCurrent[:] = M_init[:]
    N1Current[:] = N1_init[:]
    N2Current[:] = N2_init[:]
    x_YCurrent = np.ones(N) # initial x_Y
    # gCurrent = np.zeros(N)
    wCurrent = w0
    prodRat = 0

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)
    # 2D
    MArray =  np.zeros((no_of_stores,N))
    N1Array = np.zeros((no_of_stores,N))
    N2Array = np.zeros((no_of_stores,N))
    xArray = np.zeros((no_of_stores,N))
    x_YArray = np.zeros((no_of_stores,N))
    # 1D
    LArray = np.zeros(no_of_stores)
    wArray = np.zeros(no_of_stores)

    # set the values to the initial values
    MArray[0,:] =  M_init[:]
    N1Array[0,:] = N1_init[:]
    N2Array[0,:] = N2_init[:]
    xArray[0,:] = xCurrent[:]
    x_YArray[0,:] = x_YCurrent[:]
    LArray[0] = L0
    wArray[0] = wCurrent

    for count in range(1,count_max):
        MPrev[:] = MCurrent[:]
        N1Prev[:] = N1Current[:]
        N2Prev[:] = N2Current[:]
        x_YPrev[:] = x_YCurrent[:]

        # for loop to update the morphogen
        # within source
        for i in range(1,wCurrent): # iterate across space to the point before wCurrent
            MCurrent[i] = MPrev[i]*(1-(k+g0)*dt) + VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # at the index wCurrent, which has a production prodRat
        MCurrent[wCurrent] = MPrev[wCurrent]*(1-(k+g0)*dt) + prodRat*VM*dt + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[wCurrent]) * ( ( (MPrev[wCurrent+1] - MPrev[wCurrent])/(x_YPrev[wCurrent+1] + x_YPrev[wCurrent]) ) - ( (MPrev[wCurrent] - MPrev[wCurrent-1])/(x_YPrev[wCurrent-1] + x_YPrev[wCurrent]) ) ) )
        # outside of source

        for i in range(wCurrent+1,N_1):
            MCurrent[i] = MPrev[i]*(1-(k+g0)*dt) + dt*( ( 2*D / (dy**2) ) * (1/x_YPrev[i]) * ( ( (MPrev[i+1] - MPrev[i])/(x_YPrev[i+1] + x_YPrev[i]) ) - ( (MPrev[i] - MPrev[i-1])/(x_YPrev[i-1] + x_YPrev[i]) ) ) )
        
        # morphogen boundary conditions - dM/dx = 0 at x=0 and x=L
        MCurrent[0] = MPrev[0]*(1-(k+g0)*dt) + VM*dt + (D*dt/(dy**2))*(MPrev[1] - MPrev[0] )
        MCurrent[N_1] = MPrev[N_1]*(1-(k+g0)*dt) + (D*dt/(dy**2))*(MPrev[N_2] - MPrev[N_1] )

        # update x_Y and xArray
        Sx_Y = 0 # initiate Sum(x_Y) over Y, the value of x_Y at x=0 (Y=0) for all time
        for i in range(N): # iterate across space for gene products
            N1Current[i] = N1Prev[i]*(1-(k1+g0)*dt) + V1*dt*(MCurrent[i]**h/(Mstar**h + MCurrent[i]**h)) * (1 / (1 + (N2Prev[i]/N2star)**p ) )
            N2Current[i] = N2Prev[i]*(1-(k2+g0)*dt) + V2*dt / (1 + (N1Prev[i]/N1star)**q )
            x_YCurrent[i] = x_YPrev[i]/(1-g0*dt/(1+epsilon))
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
            MArray[store_index,:] = MCurrent[:]
            N1Array[store_index,:] = N1Current[:]
            N2Array[store_index,:] = N2Current[:]
            xArray[store_index,:] = xCurrent[:]
            x_YArray[store_index,:] = x_YCurrent[:]
            LArray[store_index] = LCurrent
            wArray[store_index] = wCurrent

    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, xArray, MArray, N1Array, N2Array, x_YArray, LArray, wArray


# %%%%%%%%%%%%%%%%%%% 0D solvers for N1 and N2 %%%%%%%%%%%%%%%%%%%
@numba.njit(fastmath=True)
def N1N2_0D_Growth_solver(MinputArray,gArray_at_x,V1,V2,h,p,q,k1,k2,Mstar,N1star,N2star,dt,N1_init,N2_init):
    SS_check_n1 = False
    SS_check_n2 = False
    N_time = len(MinputArray) # the number of time points in solution
    # initial conditions
    # initiate solution arrays 
    N1Array = np.zeros(N_time)
    N2Array = np.zeros(N_time)

    # set initial values
    N1Array[0] = N1_init
    N2Array[0] = N2_init
    for i in range(1,N_time):
        N1Array[i] = N1Array[i-1]*(1-(k1+gArray_at_x[i-1])*dt) + V1*dt*(MinputArray[i-1]**h/(Mstar**h + MinputArray[i-1]**h)) * (1 / (1 + (N2Array[i-1]/N2star)**p ) )
        N2Array[i] = N2Array[i-1]*(1-(k2+gArray_at_x[i-1])*dt) + V2*dt/(1 + (N1Array[i-1]/N1star)**q ) 


    if (N1Array[N_time-1] - N1Array[N_time-2])/dt < 1E-06*(V1/k1):
        SS_check_n1 = True

    if (N2Array[N_time-1] - N2Array[N_time-2])/dt < 1E-06*(V2/k2):
        SS_check_n2 = True
    # outputs
    # N1Array, N2Array, timeArray, SS_check_n1, SS_check_n2
    return N1Array, N2Array, np.arange(N_time)*dt, SS_check_n1, SS_check_n2


@numba.njit(fastmath=True)
def N1N2_0D_solver(MinputArray,V1,V2,h,p,q,k1,k2,Mstar,N1star,N2star,dt,N1_init,N2_init,SS_factor=1E-06):
    SS_check_n1 = False
    SS_check_n2 = False
    N_time = len(MinputArray) # the number of time points in solution
    # initial conditions
    # initiate solution arrays 
    N1Array = np.zeros(N_time)
    N2Array = np.zeros(N_time)
    # set initial values
    N1Array[0] = N1_init
    N2Array[0] = N2_init
    # initial conditions are 0 in this case
    for i in range(1,N_time):
        N1Array[i] = N1Array[i-1]*(1-k1*dt) + V1*dt*(MinputArray[i-1]**h/(Mstar**h + MinputArray[i-1]**h)) * (1 / (1 + (N2Array[i-1]/N2star)**p ) )
        N2Array[i] = N2Array[i-1]*(1-k2*dt) + V2*dt/(1 + (N1Array[i-1]/N1star)**q ) 


    if (N1Array[N_time-1] - N1Array[N_time-2])/(dt*N1Array[N_time-1]) < SS_factor:
        SS_check_n1 = True

    if (N2Array[N_time-1] - N2Array[N_time-2])/(dt*N2Array[N_time-1]) < SS_factor:
        SS_check_n2 = True
    
    # N1Array, N2Array, timeArray, SS_check_n1, SS_check_n2
    return N1Array, N2Array, np.arange(N_time)*dt, SS_check_n1, SS_check_n2

# %%%%%%%%%%%%%%%%%%% M tilde: the form of the morphogen for a given cell (co-moving frame) %%%%%%%%%%%%%%%%%%%

# def find_M_tilde(x0,D,L0,sRatio,VM,k,g_max,alpha,epsilon,timeArray):
def find_M_tilde_exp_decay_growth(x0,D,L0,sRatio,VM,k,g_max,alpha,epsilon,timeArray):
    # currently this function only works if a certain 'cell' is EITHER in OR out of the source - it doesn't switch.
    G = g_max / (alpha * (1 + epsilon) )
    lam = np.sqrt(D/k)
    L_ana = L0*np.exp(G*(1-np.exp(-timeArray*alpha)))
    w_ana = L_ana*sRatio
    if x0/L0 < sRatio:
        M_tilde = (VM/k) * (1 + np.sinh((w_ana-L_ana)/lam)*np.cosh(np.exp(G * (1 - np.exp(-alpha*timeArray)) ) * x0/lam)/np.sinh(L_ana/lam)) # within the source
    else:
        M_tilde = (VM/k) * (np.sinh(w_ana/lam) / np.sinh(L_ana/lam)) * np.cosh((L_ana - np.exp(G* (1 - np.exp(-alpha*timeArray)) ) * x0)/lam) # outside the source

    return M_tilde


def find_M_tilde_constant_growth(L0,g0,epsilon,sourceSize,VM,k,D,x_initial,timeArray): # x_initial is the first x value (aka comoving coordinate)
    """ A function to find the analytical solution for M tilde at a given "cell" or co-moving coordinate (starting at x_initial) over time for a tissue constantly growing at rate g0, with a fixed source size."""
    L_ana =  L0*np.exp(g0*timeArray/(1+epsilon))
    x_ana = x_initial*np.exp(g0*timeArray/(1+epsilon))
    lam = np.sqrt(D/k)
    if x_initial < sourceSize:
        # starts in source 
        index_exit_source = np.searchsorted(x_ana,sourceSize)
        M_tilde = np.zeros(len(timeArray))
        M_tilde[:index_exit_source] = (VM/k) * (1 + np.sinh((sourceSize-L_ana[:index_exit_source])/lam)*np.cosh( x_ana[:index_exit_source] /lam)/np.sinh(L_ana[:index_exit_source]/lam)) 
        M_tilde[index_exit_source:] = (VM/k) * (np.sinh(sourceSize/lam) / np.sinh(L_ana[index_exit_source:]/lam)) * np.cosh((L_ana[index_exit_source:] - x_ana[index_exit_source:] )/lam) # outside the source
    else:
        M_tilde = (VM/k) * (np.sinh(sourceSize/lam) / np.sinh(L_ana/lam)) * np.cosh((L_ana - x_ana) /lam) # outside the source
    # M_tilde_i[np.isnan(M_tilde_i)] = 0 # sometimes the values get too small
    return M_tilde

def find_M_tilde_constant_growth_linear_growing_source(L0,g0,epsilon,beta,VM,k,D,x_initial,timeArray,gamma): # x0 is the first x value (aka comoving coordinate)
    L_ana =  L0*np.exp(g0*timeArray/(1+epsilon))
    sourceSize = beta * L_ana**gamma # source size grows with the tissue size
    x_ana = x_initial*np.exp(g0*timeArray/(1+epsilon))
    lam = np.sqrt(D/k)
    source_amplitude = (VM/k)
    # for each x
    if x_initial < sourceSize[0]: # for constant growth, if a cell starts in the source it will stay in the source & vice versa 
        M_tilde = source_amplitude * (1 + np.sinh((sourceSize-L_ana)/lam)*np.cosh( (x_ana) /lam)/np.sinh(L_ana/lam)) 
    else:
        M_tilde = source_amplitude * (np.sinh(sourceSize/lam) / np.sinh(L_ana/lam)) * np.cosh((L_ana - x_ana)/lam) # outside the source
    # M_tilde_i[np.isnan(M_tilde_i)] = 0 # sometimes the values get too small
    return M_tilde

@numba.njit
def find_M_tilde_constant_growth_growing_source(L0,g0,epsilon,beta,VM,k,D,x_initial,timeArray,gamma): # x0 is the first x value (aka comoving coordinate)
    length_ev_term = np.exp(g0*timeArray/(1+epsilon))
    L_ana =  L0*length_ev_term
    sourceSize = beta * L_ana**gamma # source size grows with the tissue size
    x_ana = x_initial*length_ev_term
    lam = np.sqrt(D/k)
    source_amplitude = (VM/k)
    M_tilde = np.zeros(len(timeArray)) # initiate M tilde
    # check for each x if it is in the source
    for i in range(len(timeArray)):
        if x_ana[i] < sourceSize[i]: # if cell is within the source
            M_tilde[i] = source_amplitude * (1 + np.sinh((sourceSize[i]-L_ana[i])/lam)*np.cosh( (x_ana[i]) /lam)/np.sinh(L_ana[i]/lam))
        else:
            M_tilde[i] = source_amplitude * (np.sinh(sourceSize[i]/lam) / np.sinh(L_ana[i]/lam)) * np.cosh((L_ana[i] - x_ana[i])/lam) # outside the source
    # M_tilde_i[np.isnan(M_tilde_i)] = 0 # sometimes the values get too small
    return M_tilde

@numba.njit
def find_M_tilde_shrinking_source(L0,VM,k,D,x_val,timeArray:np.ndarray,sourceSizeArray:np.ndarray): # x0 is the first x value (aka comoving coordinate)
    lam = np.sqrt(D/k)
    # find the index at which the x_val leaves the source
    if x_val > sourceSizeArray[0]:
        index_leave_source = 0
        M_tilde = (VM/k) * (np.sinh(sourceSizeArray/lam) / np.sinh(L0/lam)) * np.cosh((L0 - x_val)/lam) # outside the source
    else:
        index_leave_source = len(sourceSizeArray) - np.searchsorted(np.flip(sourceSizeArray), x_val)
        # within source
        M_tilde = np.zeros(len(timeArray))
        M_tilde[:index_leave_source] = (VM/k) * (1 + np.sinh((sourceSizeArray[:index_leave_source]-L0)/lam)*np.cosh( (x_val) /lam)/np.sinh(L0/lam))
        M_tilde[index_leave_source:] = (VM/k) * (np.sinh(sourceSizeArray[index_leave_source:]/lam) / np.sinh(L0/lam)) * np.cosh((L0 - x_val)/lam) # outside the source
    return M_tilde, index_leave_source - 1


def find_N1_SS_given_N2(N2_values,M_value,V1,k1,h,p,Mstar,N2star):
    return (V1/k1) *(1/( (Mstar/M_value)**h + 1)) * (1 / (1 + (N2_values/N2star)**p ) )

# N1 for M -> inf
def find_N1_SS_lim_Minf(V1,k1,V2,k2,p,N2star):
    return (V1/k1) * (1 / (1 + ( V2 / (k2*N2star))**p ) )

def find_N2_SS_given_N1(N1_values,V2,k2,q,N1star):
    return (V2/k2) / (1 + (N1_values/N1star)**q )


def find_dN1dt(N1_values,N2_values,V1,k1,h,p,M_value,Mstar,N2star):
    return  V1 *(1/( (Mstar/M_value)**h + 1)) * (1 / (1 + (N2_values/N2star)**p ) ) - k1 * N1_values

def find_dN2dt(N1_values,N2_values,V2,k2,q,N1star):
    return V2 / (1 + (N1_values/N1star)**q ) - k2*N2_values

# %%%%%%%%%%%%%%%%%%% critical curve of bistability equations %%%%%%%%%%%%%%%%%%%


def find_c1star(V1,k1,N1star):
    return V1/(k1*N1star)

def find_c1(V1,k1,N1star,M_value,h,Mstar):
    return (V1/(k1*N1star))*(M_value**h/(Mstar**h + M_value**h))

def find_c2(V2,k2,N2star):
    return V2/(k2*N2star)

def pos_hill_function(M_value,h,Mstar):
    return (M_value**h/(Mstar**h + M_value**h))

def find_M_from_c1(c1,Mstar,V1,k1,N1star,h):
    """ Given a value of c1, the production rate V1, the Hill parameters N1star and h, use the definition of c1 to find the M, morphogen concentration to give this value  """
    return Mstar/ ( ( V1/ (k1*c1*N1star ) -1  )**(1/h) )

def find_critical_M_values(V1,V2,k1,k2,N1star,N2star,Mstar,q,p,h,estimate_ratio=1): 
    """ Function that finds the critical values of the morphogen M for given GRN parameterrs."""
    if q*p == 1:
        # no critical values exist
        return np.nan, np.nan, np.nan, np.nan
    c2_value = V2/(k2*N2star) # find c2, a function of the given constants
    X_split = ( (q + 1)/(q*p -1) )**((1/q))
    X_est_high = 100 # X is a normalised value. This is a high estimate to try and get the higher critical value
    min_n1_bound = 1/(q*p - 1)**(1/q) # in order for c2 to be real, X must be larger than min_n1_bound
    X_est_low = min_n1_bound*estimate_ratio # this is to try and find the value just above the minimum X bound
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
        sol2 = root_scalar(c2_parametric_critical_curve_equation_0_simple,args=(p,q,c2_value),bracket=[min_n1_bound*initial_X_ratio,X_split],x0=X_est_low)
        solution_2_converges = True 
    except ValueError:
        initial_X_ratio = 1.00000000000001
        try:
            sol2 = root_scalar(c2_parametric_critical_curve_equation_0_simple,args=(p,q,c2_value),bracket=[min_n1_bound*initial_X_ratio,X_split],x0=X_est_low)
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
    # c1 has a maximum value of c1star, aka V1/(k1*N1star)
    c1star = V1/(k1*N1star)
    if c1_critical1 > c1star:
        # print("find_critical_M_values: Higher critical value (c1critical1) not reached for any M value")
        M_critical2 = find_M_from_c1(c1_critical2,Mstar,V1,k1,N1star,h)
        return c1_critical1, c1_critical2, np.nan, M_critical2
    
    if c1_critical2 > c1star:
        # print("find_critical_M_values: Higher critical value (c1critical2) not reached for any M value")
        M_critical1 = find_M_from_c1(c1_critical1,Mstar,V1,k1,N1star,h)
        return c1_critical1, c1_critical2, M_critical1, np.nan
    
    M_critical1 = find_M_from_c1(c1_critical1,Mstar,V1,k1,N1star,h)
    M_critical2 = find_M_from_c1(c1_critical2,Mstar,V1,k1,N1star,h)
    # print(type(M_critical1))
    # print(type(c1_critical1))

    if type(M_critical1) == np.ndarray:
        # print('M_critical1')
        M_critical1 = M_critical1[0]

    if type(M_critical2) == np.ndarray:
        # print('M_critical2')
        M_critical2 = M_critical2[0]

    if type(c1_critical1) == np.ndarray:
        # print('c1_critical1')
        c1_critical1 = c1_critical1[0]

    if type(c1_critical2) == np.ndarray:
        # print('c1_critical2')
        c1_critical2 = c1_critical2[0]

    return c1_critical1, c1_critical2, M_critical1, M_critical2

def find_plot_critical_curve(q,p,max_n1_bound,no_points=1000):
    """ Function that takes the Hill coefficients and a maximum value of the normalised N1 concentration and returns n1_values, c1_values, and c2_values
        This function uses the critical c1 and c2 functions above to make it easier to plot the critical values of c1 vs c2. """
    min_n1_bound = find_min_n1_bound_for_plotting_c1_c2_parametric(q,p) 
    n1_values = np.logspace(np.log10(min_n1_bound),np.log10(max_n1_bound),no_points)[1:]
    c1_values = find_c1_parametric_critical_curve(n1_values,p,q)
    c2_values = find_c2_parametric_critical_curve(n1_values,p,q)
    return n1_values, c1_values, c2_values


def find_x_critical_from_M_critical(MCurrent,M_critical,xCurrent,N):
    """ Function that numerically finds the critical x position between stability given the critical morphogen value and a morphogen profile"""
    backwards_conc = np.flip(MCurrent)
    backwards_x = np.flip(xCurrent)
    post_index = np.searchsorted(backwards_conc,M_critical)
    if post_index == N: # if C_thr > than even C[0]
        print("find_x_critical_from_M_critical: Mcritical is too high >M(x=0) - not reached for this morphogen gradient")
        x_critical = 0
    elif post_index == 0:
        print("find_x_critical_from_M_critical: Mcritical is too low  <M(x=N) - not reached for this morphogen gradient")
        x_critical = backwards_x[0]
    else:
        pre_index = post_index - 1
        cStep = backwards_conc[post_index] - backwards_conc[pre_index] # difference in concentration
        xStep = backwards_x[post_index]- backwards_x[pre_index] # difference in space
        c_Error = backwards_conc[post_index] - M_critical # the error in concentration from the index before the threshold, the overshoot
        x_Error = xStep*c_Error/cStep
        x_critical = backwards_x[post_index] - x_Error
    if type(x_critical) == np.ndarray:
        x_critical = x_critical[0]
    return x_critical
    
# def find_bistable_region_size(sRatio,L0,VM,k,D,N,Lf,xf,V1,V2,k1,k2,N1star,N2star,Mstar,q,p,h,estimate_ratio_input=1.01):
#     sourceSize = sRatio*L0
#     m_ana_f = FuncAnalytical.find_C_ana_SS(VM,k,D,sourceSize,N,Lf,xf,ratio_given=False)
#     c1_critical1, c1_critical2, M_critical1, M_critical2 = find_critical_M_values(V1,V2,k1,k2,N1star,N2star,Mstar,q,p,h,estimate_ratio=estimate_ratio_input)
#     x_critical1 = find_x_critical_from_M_critical(m_ana_f,M_critical1,xf,N)
#     x_critical2 = find_x_critical_from_M_critical(m_ana_f,M_critical2,xf,N)
#     return m_ana_f, x_critical1, x_critical2, abs(x_critical1 - x_critical2)


def find_bifurcation_diagram(
        M_critical1,M_critical2,V1,k1,h,p,Mstar,N2star,V2,k2,q,N1star,
        lowest_M = 1e-2, highest_M = 1e5,
        estimate_ratio=1.01,num_M_val=100,number_of_points = 500,
        starting_estimate_higher = [2,-1],
        starting_estimate_unstable = [1,2], 
        starting_estimate_lower = [-4,2.5],
        n1min= 10**-4,n1max = 10**4,n2min =10**-4, n2max = 10**4,
        starting_estimate_higher_inputted = True

    ):
    if np.isnan(M_critical2):
        M_critical2 = highest_M

    # if np.isnan(M_critical1): # does this ever happen?

    unstable_M_values = np.logspace(np.log10(M_critical1*estimate_ratio), np.log10(M_critical2/estimate_ratio),num_M_val)
    upper_M_values = np.logspace(np.log10(M_critical1),np.log10(highest_M),num_M_val)
    lower_M_values = np.logspace(np.log10(lowest_M),np.log10(M_critical2),num_M_val)

    unstable_N1_solArray = np.array([])
    upper_N1_solArray = np.array([])
    lower_N1_solArray = np.array([])
    unstable_N2_solArray = np.array([])
    upper_N2_solArray = np.array([])
    lower_N2_solArray = np.array([])


    for i in range(len(unstable_M_values)):
        M_val = unstable_M_values[i]
        
        # Interpolate both nullclines
        N1_values = np.logspace(np.log10(n1min), np.log10(n1max), number_of_points)
        N2_values = np.logspace(np.log10(n2min), np.log10(n2max), number_of_points)
        N1_SS_values = find_N1_SS_given_N2(N2_values,M_val,V1,k1,h,p,Mstar,N2star)
        N2_SS_values = find_N2_SS_given_N1(N1_values,V2,k2,q,N1star) 


        f_N1_of_N2 = interp1d(np.log10(N2_values), np.log10(N1_SS_values), bounds_error=False, fill_value=np.nan)
        f_N2_of_N1 = interp1d(np.log10(N1_values), np.log10(N2_SS_values), bounds_error=False, fill_value=np.nan)
        # Define the function to find where the difference between the two is zero
        def difference_in_nullclines(logN):
            logN1, logN2 = logN
            val1 = f_N1_of_N2(logN2) - logN1
            val2 = f_N2_of_N1(logN1) - logN2
            return [val1, val2]
        unstable_logN1_sol, unstable_logN2_sol = fsolve(difference_in_nullclines, starting_estimate_unstable)
        unstable_N1_sol = 10**unstable_logN1_sol
        unstable_N2_sol = 10**unstable_logN2_sol 
        unstable_N1_solArray = np.append(unstable_N1_solArray,unstable_N1_sol)
        unstable_N2_solArray = np.append(unstable_N2_solArray,unstable_N2_sol)

    for i in range(len(upper_M_values)):
        M_val = upper_M_values[i]
        number_of_points = 500
        # high N1 solution
        N1_values = np.logspace(np.log10(n1min), np.log10(n1max), number_of_points)
        N2_values = np.logspace(np.log10(n2min), np.log10(n2max), number_of_points)
        N1_SS_values = find_N1_SS_given_N2(N2_values,M_val,V1,k1,h,p,Mstar,N2star)
        N2_SS_values = find_N2_SS_given_N1(N1_values,V2,k2,q,N1star) 
        f_N1_of_N2 = interp1d(np.log10(N2_values), np.log10(N1_SS_values), bounds_error=False, fill_value=np.nan)
        f_N2_of_N1 = interp1d(np.log10(N1_values), np.log10(N2_SS_values), bounds_error=False, fill_value=np.nan)
        
        def difference_in_nullclines(logN):
            logN1, logN2 = logN
            val1 = f_N1_of_N2(logN2) - logN1
            val2 = f_N2_of_N1(logN1) - logN2
            return [val1, val2]
        
        # N2_starting_estimate = find_N2_SS_given_N1(V1/k1,V2,k2,q,N1star)
        # starting_estimate_higher = [np.log10(V1/k1),np.log10(N2_starting_estimate)]
                
        if starting_estimate_higher_inputted == False:
            N1_amp = pos_hill_function(M_val,h) * V1/k1 
            starting_estimate_higher = [np.log10(N1_amp),np.log10(find_N2_SS_given_N1(N1_amp,V2,k2,q,N1star))]

        high_logN1_sol, high_logN2_sol = fsolve(difference_in_nullclines, starting_estimate_higher)
        # high_logN1_sol, high_logN2_sol = fsolve(difference_in_nullclines, starting_estimate)
        high_N1_sol = 10**high_logN1_sol
        high_N2_sol = 10**high_logN2_sol 
        upper_N1_solArray = np.append(upper_N1_solArray,high_N1_sol)
        upper_N2_solArray = np.append(upper_N2_solArray,high_N2_sol)

    for i in range(len(lower_M_values)):
        M_val = lower_M_values[i]
        number_of_points = 500
        # low N1 solution
        N1_values = np.logspace(np.log10(n1min), np.log10(n1max), number_of_points)
        N2_values = np.logspace(np.log10(n2min), np.log10(n2max), number_of_points)
        N1_SS_values = find_N1_SS_given_N2(N2_values,M_val,V1,k1,h,p,Mstar,N2star)
        N2_SS_values = find_N2_SS_given_N1(N1_values,V2,k2,q,N1star) 
        f_N1_of_N2 = interp1d(np.log10(N2_values), np.log10(N1_SS_values), bounds_error=False, fill_value=np.nan)
        f_N2_of_N1 = interp1d(np.log10(N1_values), np.log10(N2_SS_values), bounds_error=False, fill_value=np.nan)
        def difference_in_nullclines(logN):
            logN1, logN2 = logN
            val1 = f_N1_of_N2(logN2) - logN1
            val2 = f_N2_of_N1(logN1) - logN2
            return [val1, val2]
        
        N1_starting_estimate = find_N1_SS_given_N2(V2/k2,M_val,V1,k1,h,p,Mstar,N2star)
        starting_estimate_lower = [np.log10(N1_starting_estimate),np.log10(V2/k2)]
        low_logN1_sol, low_logN2_sol = fsolve(difference_in_nullclines, starting_estimate_lower)
        low_N1_sol = 10**low_logN1_sol
        low_N2_sol = 10**low_logN2_sol 
        lower_N1_solArray = np.append(lower_N1_solArray,low_N1_sol)
        lower_N2_solArray = np.append(lower_N2_solArray,low_N2_sol)
    

    return unstable_M_values, unstable_N1_solArray, unstable_N2_solArray, upper_M_values, upper_N1_solArray, upper_N2_solArray, lower_M_values, lower_N1_solArray, lower_N2_solArray

