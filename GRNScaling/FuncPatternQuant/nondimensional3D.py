import numpy as np
import numba
import FuncAnalyticalv2.nondimensional3D as FuncAnalytical
import FuncAnalyticalv2.dimensional as dimFuncAnalytical # for growth functions
import FuncTogglev2.nondimensional3D as FuncToggle

def find_gene_boundary(geneArrayCurrent,xCurrent,fraction,norm_global=True,norm_value=1):
    """ Function to find the position (x value) where a gene reaches a given fraction % of its maximum. """

    # find the normalisation value: either the maxmimum of geneArrayCurrent, or a given value
    if norm_global == True: # normalise by the max of the whole distribution
        normGeneArray = geneArrayCurrent/norm_value
    else: # normalise by the current maximum
        normGeneArray = geneArrayCurrent/max(geneArrayCurrent)
    
    indices_post = np.where(np.diff(np.sign(normGeneArray - fraction)))[0] # find indices just before crossings 
    if len(indices_post) != 1:  # if there are 0 or multiple crossings
        print(f'No unique gene boundary; non-monotonic gene array')
        return np.nan, 0

    index_boundary = indices_post[0]

    if index_boundary == 0:
        return np.nan, 0
    
    g_step = normGeneArray[index_boundary] - normGeneArray[index_boundary-1]
    x_step = xCurrent[index_boundary] - xCurrent[index_boundary-1]
    g_error = normGeneArray[index_boundary] - fraction 
    x_error = x_step*g_error/g_step
    # return xCurrent[index_m] - x_error - source_size, index_m # the distance of the boundary from the edge of the source
    return xCurrent[index_boundary] - x_error, index_boundary # the distance of the boundary from the edge of tissue


# @numba.njit(fastmath=True)
# def find_gene_boundaryArray(geneArray:np.ndarray,xArray:np.ndarray,sourceSize,fraction,scaledsource=False,norm_value=1):
#     """ Function to find the position (x value) where a gene reaches a given fraction % of its maximum, over a time series """
#     num_timepoints = len(geneArray)
#     gene_boundaryArray = np.zeros(num_timepoints)
#     normGeneArray = geneArray/norm_value
#     # iterate in time
#     for i in range(0,num_timepoints):
#         # check if there is 0 concentration at this timepoint
#         if max(geneArray[i,:])==0: # if there is 0 concentration
#             gene_boundaryArray[i] = 0 # set the position of the boundary to 0
#         else: # continue if there is non-zero concentration at this timepoint
#             # find the index where the normalised gene array is larger than fraction
        
#             indices_post = np.where(np.diff(np.sign(normGeneArray[i,:] - fraction)))[0] # find indices just before crossings 
#             if len(indices_post) != 1:  # if there are 0 or multiple crossings
#                 print(f'No unique gene boundary crossing for index {i}')
#                 continue # skip to next index

#             index_boundary = indices_post[0]
#             # check this index is non-zero
#             if index_boundary == 0:
#                 gene_boundary = np.nan

#             # check this index is not the far edge of the tissue
#             elif index_boundary == len(geneArray[i,:]):
#                 gene_boundary = np.nan
            
#             else: # linearly interpolate
#                 xCurrent_post = xArray[i,index_boundary] # the x position just past the boundary
#                 xCurrent_pre = xArray[i,index_boundary-1] # the x position just before the boundary
#                 # find the size of the source - update according to L if it scales
#                 if scaledsource == True: # if the source is scaled, the size of the source is proportional to L, and a ratio has been given
#                     sourceSize = sourceSize*xArray[i,-1] # update the size of the source 
                
#                 g_step = normGeneArray[index_boundary] - normGeneArray[index_boundary-1]
#                 x_step = xCurrent_post - xCurrent_pre
#                 g_error = normGeneArray[index_boundary] - fraction 
#                 x_error = x_step*g_error/g_step
#                 # gene_boundary = xCurrent_post - x_error - sourceSize # the linearly interpolated gene boundary from the edge of the source
#                 gene_boundary = xCurrent_post - x_error # the linearly interpolated gene boundary 
        
        
#             gene_boundaryArray[i] = gene_boundary
    
#     return gene_boundaryArray

# a function to find tau (the respecification time for a cell to move from the edge of the bistable region to the N2 boundary from the 1D data) )
def quantify_tau_1D(n2_boundaryArray_1D,timeArray_1D,xArray_1D,mArray_1D,n2Array_1D,m_critical1,g_max,fraction,v2,k2):
    """ A function to quantify tau from 1D simulation data. 
        Iterates over cells, finds those that leave the bistable region and become the n2 boundary and the times at which they do so.
        Returns the tau values for each cell that meets this criteria; their indices & the time they leave the bistable region & time they become the n2 boundary.

        This method is limited by the number of spatial indices of the 1D simulation.
        """

    # find the index for the cell which starts at mcritical, the edge of the bistable region (and thus immediately leaves)
    index_edge_bistable = len(mArray_1D[0,:]) - np.searchsorted(np.flip(mArray_1D[0,:]), m_critical1)

    # find the index corresponding to the boundary at final time (i.e. the cell that just becomes the N2 boundary)
    index_cell_N2_boundary_end = np.searchsorted(xArray_1D[-1,:], n2_boundaryArray_1D[-1])

    indices = np.arange(min(index_cell_N2_boundary_end,index_edge_bistable)+1,max(index_cell_N2_boundary_end,index_edge_bistable)-1,dtype=int) # all the cells that leave the bistable region and go on to become the N2 boundar
    # initiate storing arrays
    indices_tau = np.array([],dtype=int) # the spatial indices
    tau_values = np.array([])
    time_n2_boundary_values = np.array([])
    time_leave_bistable_values = np.array([])

    n2_threshold = fraction*v2/(k2+g_max)
    # iterate over trajectories for different initial positions
    if len(indices) == 0:
        print('No cells that leave the bistable region and become the N2 boundary')
        return tau_values, indices_tau, time_leave_bistable_values, time_n2_boundary_values

    for index in indices: # iterate through possible space indices
        # use MArray and mcritical to find when the cell leaves the bistable region
        mArray_at_index = mArray_1D[:,index] # slice to find M tilde for this initial position
        # find where marray = mcritical1 for general non-monotonic array
        Mcrit_crossings = np.where(np.diff(np.sign(mArray_at_index - m_critical1))) # find indices just before crossings 
        if len(Mcrit_crossings[0]) != 1:  # if there are 0 or multiple crossings
            print(f'No unique crossing of mcritical for initial position index {index}')
            continue # skip to next index

        # find the time when the cell leaves the bistable region
        index_leave_bistable = Mcrit_crossings[0][0]
        m_error = mArray_at_index[index_leave_bistable] - m_critical1
        m_step = mArray_at_index[index_leave_bistable] - mArray_at_index[index_leave_bistable-1]
        t_step_bi = timeArray_1D[index_leave_bistable] - timeArray_1D[index_leave_bistable-1]
        t_error_bi = (m_error/m_step) * t_step_bi
        time_leave_bistable = timeArray_1D[index_leave_bistable] - t_error_bi 


        n2Array_at_index = n2Array_1D[:,index]
        # assumes monotonic n2
        if n2Array_at_index[0] > n2_threshold: # cells that were already the N2 boundary 
            # print(f'Cell was already at or above N2 boundary threshold for initial position index {index}')
            continue # skip to next index
        
        # assumes that n2Array_at_index is monotonically increasing
        index_n2_enter =  np.searchsorted(n2Array_at_index,n2_threshold) # find the index when the cell hits the N2 threshold (for this growth)
        if index_n2_enter == len(n2Array_at_index): # if the index where the cell hits the N2 threshold is final / beyond
            print(f'Never reaches N2 boundary for initial position index {index}')
            continue # skip to next index

        # linear interpolation for the time to reach the n2
        n2_error = n2Array_at_index[index_n2_enter] - n2_threshold 
        n2_step = n2Array_at_index[index_n2_enter] - n2Array_at_index[index_n2_enter-1]
        t_step = timeArray_1D[index_n2_enter] - timeArray_1D[index_n2_enter-1]
        t_error = (n2_error/n2_step) * t_step
        time_n2_boundary = timeArray_1D[index_n2_enter] - t_error
        tau = time_n2_boundary - time_leave_bistable # find the time difference 
        # print(mArray_at_index[0],time_n2_boundary, time_leave_bistable, tau)
        indices_tau = np.append(indices_tau, index)
        tau_values = np.append(tau_values, tau)
        time_n2_boundary_values = np.append(time_n2_boundary_values, time_n2_boundary)
        time_leave_bistable_values = np.append(time_leave_bistable_values, time_leave_bistable)

    return tau_values, indices_tau, time_leave_bistable_values, time_n2_boundary_values

# for cells leaving the lower boundary of the bistable region; so specifying from n2 to n1
def quantify_tau_l_1D(n2_boundaryArray_1D,timeArray_1D,xArray_1D,mArray_1D,n2Array_1D,m_critical,g_max,fraction,v2,k2):
    """ A function to quantify tau from 1D simulation data. 
        Iterates over cells, finds those that leave the bistable region and become the n2 boundary and the times at which they do so.
        Returns the tau values for each cell that meets this criteria; their indices & the time they leave the bistable region & time they become the n2 boundary.

        This method is limited by the number of spatial indices of the 1D simulation.
        """

    # find the index for the cell which starts at mcritical, the edge of the bistable region (and thus immediately leaves)
    index_edge_bistable = len(mArray_1D[0,:]) - np.searchsorted(np.flip(mArray_1D[0,:]), m_critical)

    # find the index corresponding to the boundary at final time (i.e. the cell that just becomes the N2 boundary)
    index_cell_N2_boundary_end = np.searchsorted(xArray_1D[-1,:], n2_boundaryArray_1D[-1])

    indices = np.arange(min(index_cell_N2_boundary_end,index_edge_bistable)+1,max(index_cell_N2_boundary_end,index_edge_bistable)-1,dtype=int) # all the cells that leave the bistable region and go on to become the N2 boundar

    # initiate storing arrays
    indices_tau = np.array([],dtype=int) # the spatial indices
    tau_values = np.array([])
    time_n2_boundary_values = np.array([])
    time_leave_bistable_values = np.array([])

    n2_threshold = fraction*v2/(k2+g_max)

    # iterate over trajectories for different initial positions
    if len(indices) == 0:
        print('No cells that leave the bistable region and become the N2 boundary')
        return tau_values, indices_tau, time_leave_bistable_values, time_n2_boundary_values

    for index in indices: # iterate through possible space indices
        # use MArray and mcritical to find when the cell leaves the bistable region
        mArray_at_index = mArray_1D[:,index] # slice to find M tilde for this initial position
        # find where marray = mcritical for general non-monotonic array
        Mcrit_crossings = np.where(np.diff(np.sign(mArray_at_index - m_critical))) # find indices just before crossings 
        if len(Mcrit_crossings[0]) != 1:  # if there are 0 or multiple crossings
            print(f'No unique crossing of mcritical for initial position index {index}')
            continue # skip to next index

        # find the time when the cell leaves the bistable region
        index_leave_bistable = Mcrit_crossings[0][0]
        m_error = mArray_at_index[index_leave_bistable] - m_critical
        m_step = mArray_at_index[index_leave_bistable] - mArray_at_index[index_leave_bistable-1]
        t_step_bi = timeArray_1D[index_leave_bistable] - timeArray_1D[index_leave_bistable-1]
        t_error_bi = (m_error/m_step) * t_step_bi
        time_leave_bistable = timeArray_1D[index_leave_bistable] - t_error_bi 


        # find where the cell reaches the n2 threshold (i.e becomes the n2 boundary from higher n2)
        n2Array_at_index = n2Array_1D[:,index]
        # assumes monotonic n2
        if n2Array_at_index[0] < n2_threshold: # cell is not the n2_threshold
            # print(f'Cell was already at or above N2 boundary threshold for initial position index {index}')
            continue # skip to next index
        
        # assumes that n2Array_at_index is monotonically increasing
        n2_thr_crossings = np.where(np.diff(np.sign(n2Array_at_index - n2_threshold)))
        if len(n2_thr_crossings[0]) != 1:  # if there are 0 or multiple crossings
            print(f'No unique crossing of n2 threshold for initial position index {index}')
            continue # skip to next index
        

        index_n2_enter = n2_thr_crossings[0][0] # find the index just before crossing the n2 threshold 
        # linear interpolation for the time to reach the n2
        n2_error = n2Array_at_index[index_n2_enter] - n2_threshold 
        n2_step = n2Array_at_index[index_n2_enter] - n2Array_at_index[index_n2_enter-1]
        t_step = timeArray_1D[index_n2_enter] - timeArray_1D[index_n2_enter-1]
        t_error = (n2_error/n2_step) * t_step
        time_n2_boundary = timeArray_1D[index_n2_enter] - t_error
        tau = time_n2_boundary - time_leave_bistable # find the time difference 
        # print(mArray_at_index[0],time_n2_boundary, time_leave_bistable, tau)
        indices_tau = np.append(indices_tau, index)
        tau_values = np.append(tau_values, tau)
        time_n2_boundary_values = np.append(time_n2_boundary_values, time_n2_boundary)
        time_leave_bistable_values = np.append(time_leave_bistable_values, time_leave_bistable)

    return tau_values, indices_tau, time_leave_bistable_values, time_n2_boundary_values


# using generated 0D simulations, find tau, the time it takes for a cell to become the N2 boundary after leaving the bistable region
def quantify_tau_num_0D(timeArray_0D,n2Array_0D,xArray_0D,x_crit_upper,fraction,v2,k2,g0,index_leave_bistable=1):
    # can replace xArray_0D and x_crit_upper with m tilde and the m critical value
    if index_leave_bistable == 1:
        index_leave_bistable = np.searchsorted(xArray_0D,x_crit_upper) # find the index when the cell leaves the bistable region using the critical value
    
    n2_amp_eff = v2 / (k2 + g0)
    index_n2_enter =  np.searchsorted(n2Array_0D,fraction*n2_amp_eff) # find the index when the cell hits the N2 threshold (for this growth)
    # linear interpolation
    n2_error = n2Array_0D[index_n2_enter] - fraction*n2_amp_eff
    n2_step = n2Array_0D[index_n2_enter] - n2Array_0D[index_n2_enter-1]
    t_step = timeArray_0D[index_n2_enter] - timeArray_0D[index_n2_enter-1]
    t_error = (n2_error/n2_step) * t_step
    tau = timeArray_0D[index_n2_enter] - t_error - timeArray_0D[index_leave_bistable] # find the time difference 
    return tau


def quantify_tau_l_num_0D(timeArray_0D,n2Array_0D,m_tilde,mcritical,fraction,v2,k2,g0,index_leave_bistable=1):
    if index_leave_bistable == 1:
        index_leave_bistable = np.searchsorted(m_tilde,mcritical) # find the index when the cell leaves the bistable region using the critical value
    
    n2_amp_eff = v2 / (k2 + g0)
    index_n2_enter =  np.where(np.diff(np.sign(n2Array_0D - fraction*n2_amp_eff)))[0][0] # find the index when the cell hits the N2 threshold (for this growth)
    # linear interpolation
    n2_error = n2Array_0D[index_n2_enter] - fraction*n2_amp_eff
    n2_step = n2Array_0D[index_n2_enter] - n2Array_0D[index_n2_enter-1]
    t_step = timeArray_0D[index_n2_enter] - timeArray_0D[index_n2_enter-1]
    t_error = (n2_error/n2_step) * t_step
    tau = timeArray_0D[index_n2_enter] - t_error - timeArray_0D[index_leave_bistable] # find the time difference 
    return tau


# and find tau,  the time it takes for a cell to become the N2 boundary after leaving the bistable region
def find_tau_num_0D(v1,v2,k1,k2,h,p,q,vm,km,Dm,g0,epsilon,L0,sourceSize,x_critical1,fraction,store_no_0D):
    x_initial = sourceSize # start from edge of the source, assuming this is < the edge of the bistable region
    t_g = FuncAnalytical.find_t_g_constant_growth(x_initial,g0,epsilon,x_critical1*10) # define a max time 
    dt_0D = min(t_g/store_no_0D,0.1/(k1+g0))*0.1
    timeArray_0D_i = np.arange(0,t_g,dt_0D)

    # generate M tilde
    m_tilde_i = FuncToggle.find_m_tilde_constant_growth(L0,g0,epsilon,sourceSize,vm,km,Dm,x_initial,timeArray_0D_i)
    # set initial values for n1 and n2 as normalised steady state values with effective degradation + dilution
    n1_init_0D = v1 / (k1 + g0) 
    n2_init_0D = 0
    # 0D simulation
    n1Array_0D_i, n2Array_0D_i, timeArray_0D_i, SS_check_n1_0D_i, SS_check_n2_0D_i = FuncToggle.n1n2_0D_Growth_solver(m_tilde_i,g0*np.ones(len(m_tilde_i)),v1,v2,k1,k2,h,p,q,dt_0D,n1_init_0D,n2_init_0D)
    xArray_0D_i = FuncAnalytical.find_L_ana_constant_growth(x_initial,g0,epsilon,timeArray_0D_i)
    tau = quantify_tau_num_0D(timeArray_0D_i,n2Array_0D_i,xArray_0D_i,x_critical1,fraction,v2,k2,g0,index_leave_bistable=1)
    return tau


# for static xcritical1, the maximum boundary position (chi_star) is static too
def find_chi_star_ana_0D_tau_from_xcrit(x_critical1,v1,v2,k1,k2,h,p,q,vm,km,Dm,g0,epsilon,L0,sourceSize0,fraction,store_no_0D=500):
    # first generate tau from the 0D simulation
    tau_0D = find_tau_num_0D(v1,v2,k1,k2,h,p,q,vm,km,Dm,g0,epsilon,L0,sourceSize0,x_critical1,fraction,store_no_0D)
    chi_star = x_critical1 * ( np.exp( (g0 * tau_0D) / (1 + epsilon)))
    return chi_star


# def find_Delta_ana_from_tau_sg(g0, epsilon, L0, xcritArray, tau_values, time_n2_boundary_values):
#     Delta_values = xcritArray * ( np.exp( (g0 * tau_values) / (1 + epsilon)) - 1)
#     L_star_values = FuncAnalytical.find_L_ana_constant_growth(L0,g0,epsilon,time_n2_boundary_values)
#     return Delta_values, L_star_values, time_n2_boundary_values


def quantify_tau_from_m_tilde(m_tilde,m_critical1,g0,v1,v2,k1,k2,h,p,q,fraction,dt_0D,
                                        t_max_factor=7.5):
    """ A function to quantify tau for a given inputted m tilde (the morphogen signal received).
        Returns the tau value along with the full n1, n2, time arrays from the 0D simulation.
        Returns the time the cell reaches the n2 boundary and the time it leaves the bistable region for use calculating Delta.
    """
    t_max_ng = t_max_factor*(1/k1 + 1/k2)
    count_max_ng = round(t_max_ng/dt_0D) # the number of time points that will be computed - round this up
    
    # generate initial conditions by simulating to steady state for m = m_tilde[0]
    n1Array_for_IC, n2Array_for_IC, timeArray_for_IC, SS_check_n1_for_IC, SS_check_n2_for_IC = FuncToggle.n1n2_0D_Growth_solver(m_tilde[0] * np.ones(count_max_ng),g0*np.ones(count_max_ng),v1,v2,k1,k2,h,p,q,dt_0D,v1/(k1+g0),0)
    
    # solve the 0D system with M tilde as the input to find the N1 and N2 solutions in time
    n1Array_0D_i, n2Array_0D_i, timeArray_0D_i, SS_check_n1_0D_i, SS_check_n2_0D_i = FuncToggle.n1n2_0D_Growth_solver(m_tilde,g0*np.ones(len(m_tilde)),v1,v2,k1,k2,h,p,q,dt_0D,n1Array_for_IC[-1],0)

    # find when n2 = fraction*v2/(k2+g_max)
    n2_threshold = fraction*v2/(k2+g0)
    n2boundary_crossings = np.where(np.diff(np.sign(n2Array_0D_i - n2_threshold))) # find indices just before crossings 
    if len(n2boundary_crossings[0]) != 1:  # if there are 0 or multiple crossings
        tau = np.nan
        time_leave_bistable = np.nan
        time_n2_boundary = np.nan
        print('No unique crossing of n2 threshold')
        return tau, n1Array_0D_i, n2Array_0D_i, timeArray_0D_i, time_n2_boundary, time_leave_bistable

    # find the time when the cell leaves the bistable region
    index_n2_enter = n2boundary_crossings[0][0]
    # index_n2_enter =  np.searchsorted(n2Array_0D_i,fraction*v2/(k2+g0)) # find the index when the cell reaches the N2 threshold (for this growth)
    if index_n2_enter == len(n2Array_0D_i): # if the 
        tau = np.nan
        time_leave_bistable = np.nan
        time_n2_boundary = np.nan
        print('Never reaches N2 boundary')
        return tau, n1Array_0D_i, n2Array_0D_i, timeArray_0D_i, time_n2_boundary, time_leave_bistable
    else: 
        # find when M tilde leaves the bistable region
        mcrit_crossings = np.where(np.diff(np.sign(m_tilde - m_critical1))) # find indices just before crossings 
        if len(mcrit_crossings[0]) != 1:  # if there are 0 or multiple crossings
            tau = np.nan
            time_leave_bistable = np.nan
            time_n2_boundary = np.nan
            print(f'No unique crossing of mcritical')
            return tau, n1Array_0D_i, n2Array_0D_i, timeArray_0D_i, time_n2_boundary, time_leave_bistable

        # find the time when the cell leaves the bistable region
        index_leave_bistable = mcrit_crossings[0][0]
        # linear interpolation to correct error
        m_error = m_tilde[index_leave_bistable] - m_critical1 
        m_step = m_tilde[index_leave_bistable] - m_tilde[index_leave_bistable-1]
        t_step_bi = timeArray_0D_i[index_leave_bistable] - timeArray_0D_i[index_leave_bistable-1]
        t_error_bi = (m_error/m_step) * t_step_bi
        time_leave_bistable = timeArray_0D_i[index_leave_bistable] - t_error_bi 
        # linearly interpolate to correct error
        n2_error = n2Array_0D_i[index_n2_enter] - n2_threshold
        n2_step = n2Array_0D_i[index_n2_enter] - n2Array_0D_i[index_n2_enter-1]
        t_step = timeArray_0D_i[index_n2_enter] - timeArray_0D_i[index_n2_enter-1]
        t_error = (n2_error/n2_step) * t_step
        time_n2_boundary = timeArray_0D_i[index_n2_enter] - t_error
        tau = time_n2_boundary - time_leave_bistable # find the time difference
    # tauArray[i], n2_leave_bistable, n1_leave_bistable, N1Array_0D_i, n2Array_0D_i, timeArray_0D_i
    # return tau, n2Array_0D_i[index_leave_bistable], N1Array_0D_i[index_leave_bistable], N1Array_0D_i, n2Array_0D_i, timeArray_0D_i, time_n2_boundary, time_leave_bistable
    return tau, n1Array_0D_i, n2Array_0D_i, timeArray_0D_i, time_n2_boundary, time_leave_bistable



def find_xcrit_at_index(cell_indices,m_critical1,mArray_1D,xArray_1D,timeArray_1D):
    """ Function to find the value of x critical (the edge of the bistable region), when the cell leaves the bistable region, for a set of spatial cell indices.
    Also finds when this cell leaves the bistable region.
    A numerical method from the 1D simulations.
     (Where xArray_at_time[cell_indices] gives the location of the cells at a given time.) """
    
    # initialise arrays for x crit and the time each index leaves the bistable region
    xcritArray = np.zeros(len(cell_indices))
    time_leave_bistableArray = np.zeros(len(cell_indices))

    # iterate over the cell indices
    for i in range(len(cell_indices)): # cell index is the index of the cell that undergoes this trajectory
        cell_index = cell_indices[i]
        mArray_at_index = mArray_1D[:,cell_index] # slice to find M tilde at this index
        # find when this cell crosses Mcritical
        Mcrit_crossings = np.where(np.diff(np.sign(mArray_at_index - m_critical1))) # find indices just before crossings 

        index_leave_bistable_i = Mcrit_crossings[0][0] # the index in time where this cell leaves the bistable region
        # linear interpolation to find error
        M_error = mArray_at_index[index_leave_bistable_i] - m_critical1
        M_step = mArray_at_index[index_leave_bistable_i] - mArray_at_index[index_leave_bistable_i-1]
        t_step_bi = timeArray_1D[index_leave_bistable_i] - timeArray_1D[index_leave_bistable_i-1]
        t_error_bi = (M_error/M_step) * t_step_bi
        time_leave_bistableArray[i] = timeArray_1D[index_leave_bistable_i] - t_error_bi

        # find xcritical from the m profile at the time that this cell leaves the bistable region
        xcritArray[i] = FuncToggle.find_x_critical_from_m_critical(mArray_1D[index_leave_bistable_i,:],m_critical1, xArray_1D[index_leave_bistable_i,:],len(xArray_1D[index_leave_bistable_i,:]))

    return xcritArray, time_leave_bistableArray

def find_chi_ana_from_tau(g0, epsilon, L0, chi_0, x_crit_values, tau_values, time_N2_boundary_values_1D, num_points=100):
    # xcrit values for the time_N2_boundary_values_1D
    chi_star = x_crit_values * ( np.exp( (g0 * tau_values) / (1 + epsilon)))
    t_star = FuncAnalytical.find_t_g_constant_growth(chi_0,g0,epsilon,chi_star[-1]) # find how long it takes to go from chi_0 to chi_star
    t_scale = np.linspace(0,t_star,num_points)
    chi_scale = FuncAnalytical.find_L_ana_constant_growth(chi_0, g0 ,epsilon, t_scale)
    L_scale = FuncAnalytical.find_L_ana_constant_growth(L0, g0 ,epsilon, t_scale)
    L_star = FuncAnalytical.find_L_ana_constant_growth(L0,g0,epsilon,time_N2_boundary_values_1D)
    return np.append(chi_scale,np.flip(chi_star)), np.append(L_scale,np.flip(L_star))


# for cells exiting the lower 
def find_chi_ana_from_tau_l(g0, epsilon, L0, chi_0, x_crit_values, tau_values, time_N2_boundary_values_1D, num_points=100):
    # for tau_l, chi_star does not need to be flipped
    # xcrit values for the time_N2_boundary_values_1D
    chi_star = x_crit_values * ( np.exp( (g0 * tau_values) / (1 + epsilon)))
    t_star = FuncAnalytical.find_t_g_constant_growth(chi_0,g0,epsilon,chi_star[0]) # find how long it takes to go from chi_0 to chi_star
    t_scale = np.linspace(0,t_star,num_points)
    chi_scale = FuncAnalytical.find_L_ana_constant_growth(chi_0, g0 ,epsilon, t_scale)
    L_scale = FuncAnalytical.find_L_ana_constant_growth(L0, g0 ,epsilon, t_scale)
    L_star = FuncAnalytical.find_L_ana_constant_growth(L0,g0,epsilon,time_N2_boundary_values_1D)
    return np.append(chi_scale,chi_star), np.append(L_scale,L_star)

def find_chi_ana_quantify_tau_1D(m_critical1,g_max,epsilon,L0,fraction,v2,k2,timeArray_1D_IC,xArray_1D_IC,mArray_1D_IC,n2Array_1D_IC,n2_boundaryArray_1D_IC):
    tauArray_1D, indices_1D, time_leave_bistable_values, time_n2_boundary_values_1D = quantify_tau_1D(n2_boundaryArray_1D_IC,timeArray_1D_IC,xArray_1D_IC,mArray_1D_IC,n2Array_1D_IC,m_critical1,g_max,fraction,v2,k2)
    xcrit_from_xArray_func, time_leave_bistableArray_func = find_xcrit_at_index(indices_1D,m_critical1,mArray_1D_IC,xArray_1D_IC,timeArray_1D_IC)
    chi_0 = n2_boundaryArray_1D_IC[0]
    chi_ana, L_values_chi = find_chi_ana_from_tau(g_max, epsilon, L0, chi_0, xcrit_from_xArray_func, tauArray_1D, time_n2_boundary_values_1D)
    return chi_ana, L_values_chi


# source growth functions

def find_Delta_ana_from_tau_sg(g0, epsilon, L0, xcritArray, tau_values, time_n2_boundary_values):
    Delta_values = xcritArray * ( np.exp( (g0 * tau_values) / (1 + epsilon)) - 1)
    L_star_values = FuncAnalytical.find_L_ana_constant_growth(L0,g0,epsilon,time_n2_boundary_values)
    return Delta_values, L_star_values, time_n2_boundary_values


# are these used?
def find_chi_ana_from_tau_source_growth(timeArray,D,k,sourceSize0,g_max,epsilon,tau,m_critical,vm,gamma,L0):
    decay_length = np.sqrt(D/k)
    # find the source size at time = t - tau

    # change this to find the source size at time = t - tau for any growth relationship 
    sourceSize_ana_shift = dimFuncAnalytical.find_sourceSizeArray(L0,g_max,epsilon,gamma,sourceSize0,timeArray - tau)
    xc = FuncAnalytical.find_x_critical_from_m_critical_anaArray(m_critical,vm,D,k,sourceSize_ana_shift)
    return xc * np.exp(g_max * tau / (1+epsilon))

def find_Delta_from_tau_source_growth(timeArray,D,k,sourceSize0,g_max,epsilon,tau,m_critical,vm,gamma,L0):
    chi = find_chi_ana_from_tau_source_growth(timeArray,D,k,sourceSize0,g_max,epsilon,tau,m_critical,vm,gamma,L0)
    # sourceSizeArray = find_sourceSizeArray(L0,g_max,epsilon,gamma,sourceSize0,timeArray)
    sourceSizeArray = dimFuncAnalytical.find_sourceSizeArray(L0,g_max,epsilon,gamma,sourceSize0,timeArray - tau)
    xc = FuncAnalytical.find_x_critical_from_m_critical_anaArray(m_critical,vm,D,k,sourceSizeArray)
    return chi - xc

# def find_tau_num_0D_sg(v1,v2,h,p,q,k1,k2,vm,k,D,g0,epsilon,L0,beta,gamma,sourceSize,m_critical1,fraction,store_no_0D=500,time_factor=60):
#     x_initial = sourceSize
#     t_max = FuncAnalytical.find_t_g_constant_growth(x_initial,g0,epsilon,x_initial*time_factor)
#     dt_0D = min(t_max/store_no_0D,0.1/(g0))*0.01
#     timeArray_0D = np.arange(0,t_max,dt_0D)
#     m_tilde = FuncToggle.find_m_tilde_constant_growth_growing_source(L0,g0,epsilon,beta,gamma,vm,k,D,x_initial,timeArray_0D) # find m tilde for case with source growth
    
#     # find the index at which the m_tilde = m_critical1
#     if m_tilde[-1] < m_critical1 and m_tilde[0] > m_critical1: # m tilde leaves the bistable region
#         # plt.plot(timeArray-sol.root,M_tilde,color=blue_colors[i],lw='1')
#         tau_0D_sg, n1Array_0D_i, n2Array_0D_i, timeArray_0D_i, time_N2_boundary, time_leave_bistable = quantify_tau_from_m_tilde(m_tilde,m_critical1,g0,v1,v2,k1,k2,h,p,q,fraction,dt_0D)
#         # find the value of xcrit at time leave bistable 
    
#     return tau_0D_sg, time_N2_boundary, time_leave_bistable



def quantify_tau_1D_monostable(n2_boundaryArray_1D,timeArray_1D,xArray_1D,n2Array_1D,chi_0,g_max,fraction,v2,k2):
    """ A function to quantify tau from 1D simulation data. 
        Since there's no bistable region, chi0 will always be in the same place at steady state.
        Iterates over cells, finds those that grow past chi0 and become the n2 boundary and the times at which they do so.
        Returns the tau values for each cell that meets this criteria; their indices & the time they grow past chi0 & time they become the n2 boundary.

        This method is limited by the spatial indices of the 1D simulation.
        """

    # find the index for the cell is at the edge of the no growth boundary (analytical expression) at t=0
    index_chi0 = np.searchsorted(xArray_1D[0,:], chi_0)


    # find the index corresponding to the boundary at final time (i.e. the cell that just becomes the N2 boundary)
    index_cell_N2_boundary_end = np.searchsorted(xArray_1D[-1,:], n2_boundaryArray_1D[-1])

    indices = np.arange(index_cell_N2_boundary_end+1,index_chi0,dtype=int) # all the cells that grow past chi0 and go on to become the N2 boundary
    # initiate storing arrays
    indices_tau = np.array([],dtype=int) # the spatial indices
    tau_values = np.array([])
    time_n2_boundary_values = np.array([])
    time_leave_chi0_values = np.array([])

    n2_threshold = fraction*v2/(k2+g_max)
    # iterate over trajectories for different initial positions
    if len(indices) == 0:
        print('No cells that grow past chi0 and become the N2 boundary')
        return tau_values, indices_tau, time_leave_chi0_values, time_n2_boundary_values

    for index in indices: # iterate through possible indices
        # find the time at which the cell is at the x position corresponding to where the boundary would be in the no growth case (chi0)
        xArray_at_index = xArray_1D[:,index]
            # i.e. where the SS becomes classified as n2 fate not n1 fate
        index_chi0_i = np.searchsorted(xArray_at_index, chi_0)

        x_error = xArray_at_index[index_chi0_i] - chi_0
        x_step = xArray_at_index[index_chi0_i] - xArray_at_index[index_chi0_i-1]
        t_step_bi = timeArray_1D[index_chi0_i] - timeArray_1D[index_chi0_i-1]
        t_error_bi = (x_error/x_step) * t_step_bi
        time_leave_chi0 = timeArray_1D[index_chi0_i] - t_error_bi 

        n2Array_at_index = n2Array_1D[:,index]
        # assumes monotonic n2
        if n2Array_at_index[0] > n2_threshold: # cells that were already the N2 boundary 
            print(f'Cell was already at or above N2 boundary threshold for initial position index {index}')
            continue # skip to next index
        
        # assumes that n2Array_at_index is monotonically increasing
        index_n2_enter =  np.searchsorted(n2Array_at_index,n2_threshold) # find the index when the cell hits the N2 threshold (for this growth)
        if index_n2_enter == len(n2Array_at_index): # if the index where the cell hits the N2 threshold is final / beyond
            print(f'Never reaches N2 boundary for initial position index {index}')
            continue # skip to next index

        # linear interpolation for the time to reach the n2
        n2_error = n2Array_at_index[index_n2_enter] - n2_threshold 
        n2_step = n2Array_at_index[index_n2_enter] - n2Array_at_index[index_n2_enter-1]
        t_step = timeArray_1D[index_n2_enter] - timeArray_1D[index_n2_enter-1]
        t_error = (n2_error/n2_step) * t_step
        time_n2_boundary = timeArray_1D[index_n2_enter] - t_error
        tau = time_n2_boundary - time_leave_chi0 # find the time difference 
        # print(mArray_at_index[0],time_n2_boundary, time_leave_bistable, tau)
        indices_tau = np.append(indices_tau, index)
        tau_values = np.append(tau_values, tau)
        time_n2_boundary_values = np.append(time_n2_boundary_values, time_n2_boundary)
        time_leave_chi0_values = np.append(time_leave_chi0_values, time_leave_chi0)

    return tau_values, indices_tau, time_leave_chi0_values, time_n2_boundary_values


def quantify_tau_num_0D_monostable(timeArray_0D,n2Array_0D,xArray_0D,chi0,fraction,v2,k2,g0):
    n2_amp_eff = v2 / (k2 + g0)
    index_n2_enter =  np.searchsorted(n2Array_0D,fraction*n2_amp_eff) # find the index when the cell hits the n2 threshold (for this growth)

    # chi0 = position of boundary in no growth case
    index_chi0_i = np.searchsorted(xArray_0D, chi0)
    x_error = xArray_0D[index_chi0_i] - chi0
    x_step = xArray_0D[index_chi0_i] - xArray_0D[index_chi0_i-1]
    t_step_bi = timeArray_0D[index_chi0_i] - timeArray_0D[index_chi0_i-1]
    t_error_bi = (x_error/x_step) * t_step_bi
    time_leave_chi0 = timeArray_0D[index_chi0_i] - t_error_bi 

    # linear interpolation
    n2_error = n2Array_0D[index_n2_enter] - fraction*n2_amp_eff
    n2_step = n2Array_0D[index_n2_enter] - n2Array_0D[index_n2_enter-1]
    t_step = timeArray_0D[index_n2_enter] - timeArray_0D[index_n2_enter-1]
    t_error = (n2_error/n2_step) * t_step
    time_n2_boundary = timeArray_0D[index_n2_enter] - t_error
    tau = time_n2_boundary - time_leave_chi0 # find the time difference 
    return tau