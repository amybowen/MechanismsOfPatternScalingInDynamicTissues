# append paths to the modules
import numpy as np
import sys
import numba
import math

sys.path.append('/Users/bowena/Documents/GitHub/GrowthPatterningModules/')
sys.path.append('/Users/bowena/Documents/GitHub/GrowthPatterningModulesv2/')
import FuncTogglev2.nondimensional4D as nondimFuncToggle
import FuncAnalyticalv2.dimensional as dimFuncAnalytical


# dimensional
def update_conc_N_NoGrowth(MPrev,NPrev,Mstar,Nstar,Vn,kn,h,p,dt):
    return NPrev*(1-kn*dt)  + Vn*dt*( (MPrev/Mstar)**h /( 1 + (MPrev/Mstar)**h )) * ( (NPrev/Nstar)**p /( 1 + (NPrev/Nstar)**p ))

def find_dNdt(N_values,Vn,kn,M_val,Mstar,Nstar,h,p):
        return Vn * ( (M_val/Mstar)**h /( (M_val/Mstar)**h + 1)) * ( (N_values/Nstar)**p /( 1 + (N_values/Nstar)**p )) - kn*N_values



# conversion between dimensional and non-dimensional parameters
def find_nd_2d_parameters(VM, Mstar, VN, Nstar):
      vm = VM/Mstar
      vn = VN/Nstar
      return vm, vn

def find_alpha_star(VN, Nstar, kn):
      return VN / (Nstar*kn)

def find_alpha(alpha_star, m_value, h):
      return alpha_star * m_value**(h)/(1 + m_value**(h))


# non dimensional - 3nd, n = N / Nstar, m = M / Mstar, t_tilde = t*kn
# alpha = VN H(m) / km

def find_critical_alpha(p):
      """ critical alpha value for bistability """
      return p * (p-1)**((1-p)/p)

def find_m_from_alpha_nd(alpha, alpha_star, h):
      return (alpha_star/alpha - 1)**(-1/h)

def find_critical_m_3nd(p,alpha_star,h):
      """ Function to find the critical value of m for bistability, using the non-dimensionalised parameters from 2d non-dimensionalisation. """
      denom = p * (p-1)**((1-p)/p)
      fraction = alpha_star / denom
      return (fraction - 1 ) **(-1/h)


# %%% functions for stability analysis - in the _3nd case %%%
def find_g(n_values,p,alpha):
      """ Function g is such that g(n) = 0 when dndt_tilde = 0"""
      return  n_values**p - alpha*n_values**(p-1) + 1

def find_n_min(alpha, p): 
      """ n min is the value of n when g(n) is at a minumum. if g(n_min) < 0, there are 3 steady states. """
      return alpha * (p-1)/p

def find_dndt_tilde(n_values,p,alpha):
      return alpha * ( (n_values)**p /( 1 + (n_values)**p )) - n_values

def find_dndt_tilde_explicit_m(n_values,m_val,h,p,alpha_star):
      return alpha_star *( (m_val)**h /(1 + (m_val)**h )) * ( (n_values)**p /( 1 + (n_values)**p )) - n_values

def pos_hill_function(m_value,h):
    return m_value**(h)/(1 + m_value**(h))


# dim
def find_m_from_alpha(alpha, Nstar, VN, kn, h):
      return ( (VN / (Nstar*kn)) /alpha - 1)**(-1/h)


@numba.njit(fastmath=True)
def n_0D_Growth_solver(mtildeArray,gArray_at_x,vn,kn,h,p,dt,n_init,SS_factor=1E-6):
    SS_check = False
    N_time = len(mtildeArray) # the number of time points in solution

    # initiate solution arrays 
    nArray = np.zeros(N_time)

    # set initial values
    nArray[0] = n_init

    for i in range(1,N_time):
        nArray[i] = nArray[i-1]*(1-(kn+gArray_at_x[i-1])*dt) + vn*dt * ( mtildeArray[i-1]**(h)/(1 + mtildeArray[i-1]**(h) )) * (nArray[i-1]**p / (1 + nArray[i-1]**p ) )

    if (nArray[N_time-1] - nArray[N_time-2]) < nArray[N_time-1]*dt*SS_factor:
        SS_check = True

    # outputs
    # nArray, timeArray, SS_check
    return nArray, np.arange(N_time)*dt, SS_check


def quantify_tau_num_0D(timeArray_0D,nArray_0D,mtilde,m_crit,fraction,vn,kn,g0,index_leave_bistable=1):
    mcrit_crossings = np.where(np.diff(np.sign(mtilde - m_crit))) # find indices just before crossings 
    if len(mcrit_crossings[0]) != 1:  # if there are 0 or multiple crossings
        print(f'No unique crossing of mcritical for initial position index')

    # find the time when the cell leaves the bistable region
    index_leave_bistable = mcrit_crossings[0][0]
    m_error = mtilde[index_leave_bistable] - m_crit
    m_step = mtilde[index_leave_bistable] - mtilde[index_leave_bistable-1]
    t_step_bi = timeArray_0D[index_leave_bistable] - timeArray_0D[index_leave_bistable-1]
    t_error_bi = (m_error/m_step) * t_step_bi
    time_leave_bistable = timeArray_0D[index_leave_bistable] - t_error_bi 

    n_amp_eff = vn / (kn + g0)
    index_n_boundary =  len(nArray_0D) - np.searchsorted(np.flip(nArray_0D),fraction*n_amp_eff) # find the index when the cell hits the N2 threshold (for this growth)
    # linear interpolation
    n_error = nArray_0D[index_n_boundary] - fraction*n_amp_eff
    n_step = nArray_0D[index_n_boundary] - nArray_0D[index_n_boundary-1]
    t_step = timeArray_0D[index_n_boundary] - timeArray_0D[index_n_boundary-1]
    t_error = (n_error/n_step) * t_step
    tau = timeArray_0D[index_n_boundary] - t_error - time_leave_bistable # find the time difference 
    print("time leave boundary",timeArray_0D[index_n_boundary] - t_error)
    print("time leave bistable",time_leave_bistable)
    return tau


# and find tau,  the time it takes for a cell to become the N2 boundary after leaving the bistable region
def find_tau_num_0D(vn,kn,h,p,vm,km,Dm,g0,epsilon,L0,sourceSize,x_critical1,m_crit,fraction,store_no_0D=500):
    x_initial = sourceSize # start from edge of the source, assuming this is < the edge of the bistable region
    t_g = dimFuncAnalytical.find_t_g_constant_growth(x_initial,g0,epsilon,x_critical1*10) # define a max time 
    dt_0D = min(t_g/store_no_0D,0.1/(kn+g0))*0.1
    timeArray_0D_i = np.arange(0,t_g,dt_0D)

    # generate M tilde
    m_tilde_i = nondimFuncToggle.find_m_tilde_constant_growth(L0,g0,epsilon,sourceSize,vm,km,Dm,x_initial,timeArray_0D_i)
    # set initial values for n1 and n2 as normalised steady state values with effective degradation + dilution
    # 0D simulation
    n_init_0D = vn / (kn + g0) 
    nArray_0D_i, timeArray_0D_i, SS_check = n_0D_Growth_solver(m_tilde_i,g0*np.ones(len(m_tilde_i)),vn,kn,h,p,dt_0D,n_init_0D)

    xArray_0D_i = dimFuncAnalytical.find_L_ana_constant_growth(x_initial,g0,epsilon,timeArray_0D_i)
    # tau = quantify_tau_num_0D(timeArray_0D_i,nArray_0D_i,xArray_0D_i,x_critical1,fraction,vn,kn,g0,index_leave_bistable=1)
    tau = quantify_tau_num_0D(timeArray_0D_i,nArray_0D_i,m_tilde_i,m_crit,fraction,vn,kn,g0,index_leave_bistable=1)

    return tau

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# for static xcritical1, the maximum boundary position (chi_star) is static too
def find_chi_star_ana_0D_tau_from_xcrit(x_critical1,m_crit,vn,kn,h,p,vm,km,Dm,g0,epsilon,L0,sourceSize0,fraction,store_no_0D=500):
    # first generate tau from the 0D simulation
    tau_0D = find_tau_num_0D(vn,kn,h,p,vm,km,Dm,g0,epsilon,L0,sourceSize0,x_critical1,m_crit,fraction,store_no_0D)
    print(tau_0D)
    chi_star = x_critical1 * ( np.exp( (g0 * tau_0D) / (1 + epsilon)))
    return chi_star


def quantify_tau_1D(n_boundaryArray_1D,timeArray_1D,xArray_1D,mArray_1D,nArray_1D,m_critical,g_max,fraction,vn,kn):
    """" a function to find tau (the respecification time for a cell to move from the edge of the bistable region to the N2 boundary from the 1D data) ) """
    # works for monotonic or non-monotonic mArray_at_index

    # find the index for the cell which starts at mcritical, the edge of the bistable region (and thus immediately leaves)
    # index_edge_bistable = len(mArray_1D[0,:]) - np.searchsorted(np.flip(mArray_1D[0,:]), m_critical)
    index_chi_0 = np.searchsorted(xArray_1D[0,:], n_boundaryArray_1D[0]) # the index that is the N2 boundary at t=0 - this is the max index

    # find the index corresponding to the boundary at final time (i.e. the cell that just becomes the N2 boundary)
    index_cell_n_boundary_end = np.searchsorted(xArray_1D[-1,:], n_boundaryArray_1D[-1])
    index_cell_xc_f = len(mArray_1D[-1,:]) - np.searchsorted(np.flip(mArray_1D[-1,:]), m_critical )


    # indices = np.arange(index_cell_n_boundary_end+1,index_edge_bistable,dtype=int) # all the cells that leave the bistable region and go on to become the N2 boundary
    indices = np.arange(index_cell_xc_f+1,index_chi_0,dtype=int) # all the cells that leave the bistable region and go on to become the N2 boundary

    indices_tau = np.array([],dtype=int) # initiate an array to store indices where we can find tau
    tau_values = np.array([])
    time_n_boundary_values = np.array([])
    # iterate over trajectories for different initial positions
    for index in indices: # iterate through possible indices
        # use MArray and mcritical to find when the cell leaves the bistable region
        mArray_at_index = mArray_1D[:,index] # slice to find M tilde for this initial position
        # find where marray = mcritical1 for general non-monotonic array
        mcrit_crossings = np.where(np.diff(np.sign(mArray_at_index - m_critical))) # find indices just before crossings 
        if len(mcrit_crossings[0]) != 1:  # if there are 0 or multiple crossings
            print(f'No unique crossing of mcritical for initial position index {index}')
            continue # skip to next index

        # find the time when the cell leaves the bistable region
        index_leave_bistable = mcrit_crossings[0][0]
        m_error = mArray_at_index[index_leave_bistable] - m_critical
        m_step = mArray_at_index[index_leave_bistable] - mArray_at_index[index_leave_bistable-1]
        t_step_bi = timeArray_1D[index_leave_bistable] - timeArray_1D[index_leave_bistable-1]
        t_error_bi = (m_error/m_step) * t_step_bi
        time_leave_bistable = timeArray_1D[index_leave_bistable] - t_error_bi 

        nArray_at_index = nArray_1D[:,index]
        # assumes monotonic n2
        if nArray_at_index[0] < fraction*vn/(kn+g_max): # cells that were already the N2 boundary 
            print(f'Cell was already below n boundary threshold for initial position index {index}')
            continue # skip to next index
        
        # assumes that N2Array_at_index is monotonically increasing
        n_boundary_crossings = np.where(np.diff(np.sign(nArray_at_index - fraction*vn/(kn+g_max)))) # find indices just before crossings 
        if len(n_boundary_crossings[0]) != 1:  # if there are 0 or multiple crossings
            print(f'Never reaches N2 boundary for initial position index {index}')
            continue # skip to next index

        
        index_n_boundary = n_boundary_crossings[0][0] # the index in time where this cell leaves the bistable region
        # linear interpolation
        n2_error = nArray_at_index[index_n_boundary] - fraction*vn/(kn+g_max) 
        n2_step = nArray_at_index[index_n_boundary] - nArray_at_index[index_n_boundary-1]
        t_step = timeArray_1D[index_n_boundary] - timeArray_1D[index_n_boundary-1]
        t_error = (n2_error/n2_step) * t_step
        time_N2_boundary = timeArray_1D[index_n_boundary] - t_error
        tau = time_N2_boundary - time_leave_bistable # find the time difference 
        # print(mArray_at_index[0],time_N2_boundary, time_leave_bistable, tau)
        tau_values = np.append(tau_values, tau)
        time_n_boundary_values = np.append(time_n_boundary_values, time_N2_boundary)
        indices_tau = np.append(indices_tau, index)

    return tau_values, indices_tau, time_n_boundary_values


def find_chi_ana_from_tau(g0, epsilon, L0, chi_0, x_crit_values, tau_values, time_N2_boundary_values_1D, num_points=100):
    chi_star = x_crit_values * ( np.exp( (g0 * tau_values) / (1 + epsilon)))
    t_star = dimFuncAnalytical.find_t_g_constant_growth(chi_0,g0,epsilon,chi_star[-1]) # find how long it takes to go from chi_0 to chi_star
    t_scale = np.linspace(0,t_star,num_points)
    chi_scale = dimFuncAnalytical.find_L_ana_constant_growth(chi_0, g0 ,epsilon, t_scale)
    L_scale = dimFuncAnalytical.find_L_ana_constant_growth(L0, g0 ,epsilon, t_scale)
    L_star = dimFuncAnalytical.find_L_ana_constant_growth(L0,g0,epsilon,time_N2_boundary_values_1D)
    return np.append(chi_scale,np.flip(chi_star)), np.append(L_scale,np.flip(L_star))



# %%%%%%%%%%%%%%%%%%% 1D solvers for n %%%%%%%%%%%%%%%%%%%
@numba.njit
def n_solver_NoGrowth_fixedm_2nd(N,dt,vn,kn,p,h,store_no,n_init:np.ndarray,m_init:np.ndarray,count_max):
    """ A solver for n (self activating node, normalised) with a fixed normalised morphogen profile m over time with no growth.
        Essentially a set of 0D solvers. 
                Variables:
                N = number of spatial points
                dt = dimensional time step
                vn = non dimensional production rate of n = VN / Nstar
                kn = dimensional degradation rate of n
                p = Hill coefficient for self activation
                h = Hill coefficient for morphogen activation
                store_no = number of time steps between storing profiles
                n_init = initial condition array for n
                m_init = fixed normalised morphogen profile array for m
                count_max = total number of time steps to simulate.

                Returns:
                timeArray = array of time points at which profiles were stored
                nArray = array of normalised gene product profiles over time (n = N / Nstar)
        """
    
    # dt is dimensional time step
    
    # initiate initial conditions
    nCurrent = np.zeros(N)

    # initiate previous arrays for n
    nPrev = np.zeros(N)

    # set the current Arrays as the initial conditions
    nCurrent[:] = n_init[:]

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)

    nArray = np.zeros((no_of_stores,N))

    # set the values to the initial values
    nArray[0,:] = n_init[:]

    for count in range(1,count_max):
        nPrev[:] = nCurrent[:]

        for i in range(N): # iterate across space for gene products
            nCurrent[i] = nPrev[i]* (1-kn*dt) + vn*dt*((m_init[i])**(h)/(1 + (m_init[i])**(h) ) ) * (nPrev[i])**p  / (1 + (nPrev[i])**p ) 
        
        # store solutions every store_no
        if count % store_no == 0:
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            store_index = int(count/store_no)
            nArray[store_index,:] = nCurrent[:]


    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, nArray



# %%%%%%%%%%%%%%%%%%% 1D solvers for n %%%%%%%%%%%%%%%%%%%
@numba.njit
def N_solver_NoGrowth_fixedm(N,dt,VN,kn,p,h,Mstar,Nstar,store_no,N_init:np.ndarray,M_init:np.ndarray,count_max):
    """ A solver for n (self activating node, normalised) with a fixed normalised morphogen profile m over time with no growth.
        Essentially a set of 0D solvers. 
                Variables:
                N = number of spatial points
                dt = dimensional time step
                VN = dimensional production rate of n
                kn = dimensional degradation rate of n
                p = Hill coefficient for self activation
                h = Hill coefficient for morphogen activation
                Mstar = critical morphogen concentration for activation of N production
                Nstar = critical N concentration for activation of N production
                store_no = number of time steps between storing profiles
                N_init = initial condition array for N
                M_init = fixed normalised morphogen profile array for M
                count_max = total number of time steps to simulate.

                Returns:
                timeArray = array of time points at which profiles were stored
                NArray = array of gene product profiles over time 
        """
    
    # dt is dimensional time step
    
    # initiate initial conditions
    NCurrent = np.zeros(N)

    # initiate previous arrays for n
    NPrev = np.zeros(N)

    # set the current Arrays as the initial conditions
    NCurrent[:] = N_init[:]

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)

    NArray = np.zeros((no_of_stores,N))

    # set the values to the initial values
    NArray[0,:] = N_init[:]

    for count in range(1,count_max):
        NPrev[:] = NCurrent[:]

        for i in range(N): # iterate across space for gene products
            NCurrent[i] = NPrev[i]* (1-kn*dt) + VN*dt*(( (M_init[i]/Mstar) )**(h)/(1 + (M_init[i]/Mstar)**(h) ) ) * (NPrev[i]/Nstar)**p  / (1 + (NPrev[i]/Nstar)**p ) 
        
        # store solutions every store_no
        if count % store_no == 0:
            # store the morphogen profile C, the positions x, the length L, and the growth profile g
            store_index = int(count/store_no)
            NArray[store_index,:] = NCurrent[:]


    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, NArray


# growing solver that doesn't scale the source
@numba.njit
def n_solver_ConstantGrowth_fixed_source_2nd(N,N_1,N_2,w0,Dm,km,vm,epsilon,sourceSize,dt,dy,vn,kn,h,p,L0,store_no,m_init:np.ndarray,n_init:np.ndarray,count_max,g):
# def n_solver_NoGrowth_fixedm_2nd(N,dt,VN,kn,p,h,store_no,n_init:np.ndarray,m_init:np.ndarray,count_max):

    # ndFuncToggle.n1n2m_solver_ConstantGrowth_fixed_source(N,N_1,N_2,w0,D,k,vm,epsilon,sourceSize,dt,dy,v1,v2,k1,k2,h,p,q,L0,store_no,m_init,n1_init,n2_init,count_max,g_max)
    """ A solver for m and n over time with constant growth. The morphogen source size is fixed. """ 

    # initialise Current and Prev arrays for M, n1, n2, x_Y
    mCurrent = np.zeros(N)
    nCurrent = np.zeros(N)

    mPrev  = np.zeros(N)
    nPrev = np.zeros(N)
    x_YPrev = np.zeros(N)

    # set initial conditions as current arrays
    xCurrent = np.linspace(0,L0,N) #The co-moving coordinates: N spatial points, includes the point 0 and L0
    mCurrent[:] = m_init[:]
    nCurrent[:] = n_init[:]
    x_YCurrent = np.ones(N) # initial x_Y
    wCurrent = w0
    prodRat = 0 # production ratio between the last cell in the source and first cell out, to interpolate

    # initialise saving arrays
    no_of_stores = math.ceil(count_max/store_no)
    # 2D
    mArray =  np.zeros((no_of_stores,N))
    nArray = np.zeros((no_of_stores,N))
    xArray = np.zeros((no_of_stores,N))
    x_YArray = np.zeros((no_of_stores,N))
    # 1D
    LArray = np.zeros(no_of_stores)
    wArray = np.zeros(no_of_stores)

    # set the values to the initial values
    mArray[0,:] =  m_init[:]
    nArray[0,:] = n_init[:]
    xArray[0,:] = xCurrent[:]
    x_YArray[0,:] = x_YCurrent[:]
    LArray[0] = L0
    wArray[0] = wCurrent

    for count in range(1,count_max):
        mPrev[:] = mCurrent[:]
        nPrev[:] = nCurrent[:]
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
            # print(nPrev[i], (1-(kn+g)*dt), vn*dt*((m_init[i])**(h)/(1 + (m_init[i])**(h) ) ) * ((nPrev[i])**p/ (1 + (nPrev[i])**p ) ))
            nCurrent[i] = nPrev[i]* (1-(kn+g)*dt) + vn*dt*((mPrev[i])**(h)/(1 + (mPrev[i])**(h) ) ) * ((nPrev[i])**p/ (1 + (nPrev[i])**p ) )
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
            nArray[store_index,:] = nCurrent[:]
            xArray[store_index,:] = xCurrent[:]
            x_YArray[store_index,:] = x_YCurrent[:]
            LArray[store_index] = LCurrent
            wArray[store_index] = wCurrent

    timeArray = np.linspace(0,(count_max-1)*dt,math.ceil(count_max/store_no))
    return timeArray, xArray, mArray, nArray, x_YArray, LArray, wArray


# %%%%%%%%%%%%%%%%%%% 0D solvers for n %%%%%%%%%%%%%%%%%%%

@numba.njit(fastmath=True)
def n_0D_Growth_solver(mtildeArray,gArray_at_x,vn,kn,h,p,dt,n_init,SS_factor=1E-6):
    SS_check = False
    N_time = len(mtildeArray) # the number of time points in solution
    # initiate solution arrays 
    nArray = np.zeros(N_time)

    # set initial values
    nArray[0] = n_init
    for i in range(1,N_time):
        nArray[i] = nArray[i-1]*(1-(kn+gArray_at_x[i-1])*dt) + vn*dt * ( mtildeArray[i-1]**(h)/(1 + mtildeArray[i-1]**(h) )) * (1 / (1 + (nArray[i-1])**p ) )


    if (nArray[N_time-1] - nArray[N_time-2]) < nArray[N_time-1]*dt*SS_factor:
        SS_check = True

    # outputs
    # nArray, timeArray, SS_check
    return nArray, np.arange(N_time)*dt, SS_check


def calculate_useful_simulation_parameters(N,L0,dt_factor,D,k,g_max,kn,sourceSize,t_max,store_no):
    # calculate values
    N_1 = N-1
    N_2 = N-2
    dy = L0/(N-1) # space step
    w0 = round(sourceSize/dy)+1 # the index for the edge of the source
    dt = dt_factor*(2/(4*D/(dy**2) + max(g_max,k,kn)))   # setting dt according to k and g_max, via the vN stability criterion
    count_max = round(t_max/dt) # the number of time points that will be computed - round this up
    dt_store = dt * store_no
    return N_1, N_2, dy, w0, dt, count_max, dt_store


def calculate_useful_simulation_parameters_no_growth(N,L0,dt_factor,D,k,kn,sourceSize,store_no_ng,t_max_factor = 10):
    N_1 = N-1
    N_2 = N-2
    dy = L0/(N-1) # space step
    w0 = round(sourceSize/dy)+1 # the index for the edge of the source
    dt_ng = dt_factor*(2/(4*D/(dy**2) + max(k,kn)))   # setting dt according to k and g_max, via the vN stability criterion
    t_max_ng = t_max_factor*(1/kn)
    count_max_ng = round(t_max_ng/dt_ng) # the number of time points that will be computed - round this up
    dt_store_ng = dt_ng * store_no_ng
    return N_1, N_2, dy, w0, dt_ng, count_max_ng, dt_store_ng


def von_Neumann_and_growth_check(dt, D, dy, k, g_max, kn, epsilon):
    """ A function to perform the von Neumann stability and growth stability checks to ensure numerical stability for the concentrations and growth. """
    # %%%%%%%%%%%% STABILITY CHECK - von Neumann and growth %%%%%%%%%%%%
    stability = dt * (4*D/((dy)**2) + k+g_max)
    # print('diffusion part of error',4*D/((dy)**2))
    # print('k+g_max part of error',k+g_max)
    gStability = g_max*dt/(1+epsilon)
    print("growth stability",gStability)
    print("vN",stability)
    nstability = kn*dt
    if stability < 2 and gStability <1 and nstability < 1:
        print('Stable')
    else:
        print('Unstable')
    return


# using generated 0D simulations, find tau, the time it takes for a cell to become the N2 boundary after leaving the bistable region
def quantify_tau_num_0D(timeArray_0D,nArray_0D,xArray_0D,x_crit,fraction,vn,kn,g0,index_leave_bistable=1):
    if index_leave_bistable == 1:
        index_leave_bistable = np.searchsorted(xArray_0D,x_crit) # find the index when the cell leaves the bistable region using the critical value
    
    n_amp_eff = vn / (kn + g0)

    index_n_enter =  np.where(nArray_0D < fraction*n_amp_eff)[0] # find the index when the cell hits the threshold (for this growth)
    # linear interpolation
    n_error = nArray_0D[index_n_enter[0]] - fraction*n_amp_eff
    n_step = nArray_0D[index_n_enter[0]] - nArray_0D[index_n_enter[0]-1]
    t_step = timeArray_0D[index_n_enter[0]] - timeArray_0D[index_n_enter[0]-1]
    t_error = (n_error/n_step) * t_step
    tau = timeArray_0D[index_n_enter[0]] - t_error - timeArray_0D[index_leave_bistable] # find the time difference 
    return tau
