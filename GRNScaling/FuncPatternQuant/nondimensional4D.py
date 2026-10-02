import numpy as np
import numba
import FuncAnalyticalv2.nondimensional4D as FuncAnalytical
import FuncTogglev2.nondimensional4D as FuncToggle


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



# using generated 0D simulations, find tau, the time it takes for a cell to become the N2 boundary after leaving the bistable region
def quantify_tau_num_0D(time_ndArray_0D,n2Array_0D,xArray_0D,x_crit_upper,fraction,alpha2,nu2,g_nd,index_leave_bistable=1):
    if index_leave_bistable == 1:
        index_leave_bistable = np.searchsorted(xArray_0D,x_crit_upper) # find the index when the cell leaves the bistable region using the critical value
    
    n2_amp_eff = alpha2 / (nu2 + g_nd)
    index_N2_enter =  np.searchsorted(n2Array_0D,fraction*n2_amp_eff) # find the index when the cell hits the N2 threshold (for this growth)
    # linear interpolation
    n2_error = n2Array_0D[index_N2_enter] - fraction*n2_amp_eff
    n2_step = n2Array_0D[index_N2_enter] - n2Array_0D[index_N2_enter-1]
    t_step = time_ndArray_0D[index_N2_enter] - time_ndArray_0D[index_N2_enter-1]
    t_error = (n2_error/n2_step) * t_step
    tau = time_ndArray_0D[index_N2_enter] - t_error - time_ndArray_0D[index_leave_bistable] # find the time difference 
    return tau


# and find tau,  the time it takes for a cell to become the N2 boundary after leaving the bistable region
def find_tau_num_0D(alpha1,alpha2,nu2,h,p,q,vM_nd,k_nd,D_nd,g_nd,epsilon,L0,sourceSize,x_critical1,fraction,store_no_0D):
    x_initial = sourceSize # start from edge of the source, assuming this is < the edge of the bistable region
    t_g = FuncAnalytical.find_t_nd_g_constant_growth(x_initial,g_nd,epsilon,x_critical1*10) # define a max time 
    dt_0D = min(t_g/store_no_0D,0.1/(1+g_nd))*0.1
    timeArray_0D_i = np.arange(0,t_g,dt_0D)
    # generate M tilde
    
    m_tilde_i = FuncToggle.find_m_tilde_constant_growth(L0,g_nd,epsilon,sourceSize,vM_nd,k_nd,D_nd,x_initial,timeArray_0D_i)
    # set initial values for n1 and n2 as normalised steady state values with effective degradation + dilution
    n1_init_0D = alpha1 / (1 + g_nd) 
    n2_init_0D = 0
    # 0D simulation
    n1Array_0D_i, n2Array_0D_i, timeArray_0D_i, SS_check_n1_0D_i, SS_check_n2_0D_i = FuncToggle.n1n2_0D_Growth_solver(m_tilde_i,g_nd*np.ones(len(m_tilde_i)),alpha1,alpha2,h,p,q,nu2,dt_0D,n1_init_0D,n2_init_0D)
    xArray_0D_i = FuncAnalytical.find_L_ana_constant_growth(x_initial,g_nd,epsilon,timeArray_0D_i)
    tau = quantify_tau_num_0D(timeArray_0D_i,n2Array_0D_i,xArray_0D_i,x_critical1,fraction,alpha2,nu2,g_nd,index_leave_bistable=1)
    return tau

# static xcritical1 - the maximum boundary position (chi_star) is static too
def find_chi_star_ana_0D_tau_from_xcrit(x_critical1,alpha1,alpha2,nu2,h,p,q,vM_nd,k_nd,D_nd,g_nd,epsilon,L0,sourceSize0,fraction,store_no_0D=500):
    # first generate tau from the 0D simulation
    tau_0D = find_tau_num_0D(alpha1,alpha2,nu2,h,p,q,vM_nd,k_nd,D_nd,g_nd,epsilon,L0,sourceSize0,x_critical1,fraction,store_no_0D)
    chi_star = x_critical1 * ( np.exp( (g_nd * tau_0D) / (1 + epsilon)))
    return chi_star