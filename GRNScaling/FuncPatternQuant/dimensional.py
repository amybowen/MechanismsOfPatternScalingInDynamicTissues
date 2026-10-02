#  a module containing the quantification of the patterns: decay length, threshold concentration

import numpy as np
import matplotlib.pyplot as plt # plotting library
from scipy.optimize import curve_fit
import scipy.integrate
import dill as pickle
# import FuncAnalytical # module with the analytical equations
import FuncFit
import FuncAnalyticalv2.dimensional as FuncAnalytical
import warnings
import numba
import FuncTogglev2.dimensional as FuncToggle
from scipy.optimize import OptimizeWarning
# warnings.filterwarnings("error")

# functions to quantify the pattern
def find_x_thr(N,sRatio,C_thr,cCurrent,xCurrent,LCurrent,wCurrent): # find the distance from the edge of the source x_thr where the concentration is below Z
    backwards_conc = np.flip(cCurrent)
    backwards_x = np.flip(xCurrent)
    post_index = np.searchsorted(backwards_conc,C_thr)
    sSizeIdeal = sRatio*LCurrent
    sSizeIdeal = xCurrent[wCurrent]
    if post_index == N: # if C_thr > than even C[0]
        return -sSizeIdeal
    elif post_index == 0:
        return backwards_x[0] - sSizeIdeal
    else:
        pre_index = post_index - 1
        cStep = backwards_conc[post_index] - backwards_conc[pre_index] # difference in concentration
        xStep = backwards_x[pre_index]- backwards_x[post_index] # difference in space
        c_Error = backwards_conc[post_index] - C_thr # the error in concentration from the index before the threshold, the overshoot
        x_Error = xStep*c_Error/cStep
        x_thr = backwards_x[post_index]+x_Error - sSizeIdeal # take away the size of the source as x_thr is the distance from the edge of the source
        return x_thr

def find_x_thrArray(N,sRatio,C_thr,cArray,xArray,LArray): # find the distance from the edge of the source x_thr where the concentration is below Z
    xthrArray = []
    for index in range(len(cArray)):
        cCurrent = cArray[index,:]
        xCurrent = xArray[index,:]
        backwards_conc = np.flip(cCurrent)
        backwards_x = np.flip(xCurrent)
        post_index = np.searchsorted(backwards_conc,C_thr)
        sSizeIdeal = sRatio*LArray[index]
        if post_index == N: # if C_thr > than even C[0]
            x_thr = -sSizeIdeal
        elif post_index == 0:
            x_thr = backwards_x[0] - sSizeIdeal
        else:
            pre_index = post_index - 1
            cStep = backwards_conc[post_index] - backwards_conc[pre_index] # difference in concentration
            xStep = backwards_x[pre_index]- backwards_x[post_index] # difference in space
            c_Error = backwards_conc[post_index] - C_thr # the error in concentration from the index before the threshold, the overshoot
            x_Error = xStep*c_Error/cStep
            x_thr = backwards_x[post_index]+x_Error -sSizeIdeal # take away the size of the source as x_thr is the distance from the edge of the source
        xthrArray = np.append(xthrArray,x_thr)
    return xthrArray

def find_x_thr_source_size(N,sSize,Z,cCurrent,xCurrent): # find x_thr for a variable source size
    for i in range(N): # iterate across space
        if cCurrent[i] < Z:
            post_index = i
            # print(i)
            break
    # so the two nearest indices are post_index and pre_index
    pre_index = post_index - 1
    cStep = cCurrent[pre_index] - cCurrent[post_index] # difference in concentration
    xStep = xCurrent[post_index]- xCurrent[pre_index] # difference in space
    c_Error = cCurrent[pre_index] - Z # the error in concentration from the index before the threshold
    x_Error = xStep*c_Error/cStep
    x_thr = xCurrent[pre_index]+x_Error- sSize # take away the size of the source as x_thr is the distance from the edge of the source
    print(x_thr)
    return x_thr

# def find_decay_length_polyfit(sRatio,xCurrent,cCurrent,LCurrent,wCurrent): # find the decay length when fitting an exponentially decaying function to the morphogen profile
#     sSizeIdeal = sRatio*LCurrent # the size of the source in x-space
#     xLShift = xCurrent - sSizeIdeal
#     # fitting to cCurrent = c0 e^(-x/lamNum)
#     logc = np.log(cCurrent[wCurrent:len(cCurrent)])
#     p = np.polyfit(xLShift[wCurrent:len(cCurrent)],logc,1)
#     c0 = np.exp(p[1])
#     decay_length = -1/p[0]
#     return decay_length, c0

def find_decay_length(xCurrent,cCurrent,LCurrent,wCurrent,N,phi_sugg,c_prefactor_sugg): # find the decay length when fitting an exponentially decaying function to the morphogen profile
    try:
        popt_cosh, pcov_cosh = curve_fit(FuncFit.cosh_func,xCurrent[wCurrent:N]/LCurrent,cCurrent[wCurrent:N],p0=[c_prefactor_sugg,phi_sugg],maxfev=1000000)
        c0_cosh = popt_cosh[0]
        phi = popt_cosh[1]
        decay_length = abs(phi*LCurrent)
        pcov_lam = pcov_cosh[1,1]
        pcov_pref = pcov_cosh[0,0]
    except OptimizeWarning:
        decay_length = 0
        c0_cosh = 0
        pcov_lam = 0
        pcov_pref = 0

    return decay_length, c0_cosh, pcov_lam, pcov_pref


def find_decay_length_array(xArray,cArray,wArray,N,sMag,D,k,L0,sRatio): # find the decay length at each time point given position and conc arrays
    # L0 is only for suggestion
    decay_lengthArray = [0]
    prefactorArray = [0]
    pcov_decay_lengthArray = [0]
    pcov_prefactorArray = [0]
    lam_real = np.sqrt(D/(k))
    phi_sugg = lam_real/L0
    c_prefactor_sugg = (sMag/k) * (np.sinh(L0*sRatio/lam_real) / np.sinh(L0/lam_real))
    failed_fits = []
    for i in range(1,len(xArray)): # iterate through time
        cCurrent = cArray[i,:]
        xCurrent = xArray[i,:]
        LCurrent = xCurrent[-1]
        w = int(wArray[i])
        # gCurrent = gArray[i,w]

        try:
            popt_cosh, pcov_cosh = curve_fit(FuncFit.cosh_func,xCurrent[w:N]/LCurrent,cCurrent[w:N],p0=[c_prefactor_sugg,phi_sugg],maxfev=1000000)
            c0_cosh = popt_cosh[0]
            phi = popt_cosh[1]
            lam_cosh_fit = phi*LCurrent
            decay_lengthArray = np.append(decay_lengthArray,abs(lam_cosh_fit))
            prefactorArray = np.append(prefactorArray,c0_cosh)
            pcov_decay_lengthArray = np.append(pcov_decay_lengthArray,pcov_cosh[1,1])
            pcov_prefactorArray = np.append(pcov_prefactorArray,pcov_cosh[0,0])

            # set new suggestions for phi and c_prefactor
            phi_sugg = phi
            c_prefactor_sugg = c0_cosh
        except (RuntimeWarning,OptimizeWarning):
            decay_lengthArray = np.append(decay_lengthArray,0)
            prefactorArray = np.append(prefactorArray,0)
            pcov_decay_lengthArray = np.append(pcov_decay_lengthArray,0)
            pcov_prefactorArray = np.append(pcov_prefactorArray,0)
            failed_fits = np.append(failed_fits,i)
        

    
    return prefactorArray, decay_lengthArray, failed_fits



def find_decay_length_array_sugg(xArray,cArray,wArray,N,sMag,D,k,L0,sRatio): # find the decay length at each time point given position and conc arrays
    # L0 is only for the suggestion
    decay_lengthArray = [0]
    prefactorArray = [0]
    pcov_decay_lengthArray = [0]
    pcov_prefactorArray = [0]
    lam_real = np.sqrt(D/(k))
    phi_sugg = lam_real/L0
    c_prefactor_sugg = (sMag/k) * (np.sinh(L0*sRatio/lam_real) / np.sinh(L0/lam_real))
    failed_fits = []
    for i in range(1,len(xArray)): # iterate through time
        cCurrent = cArray[i,:]
        xCurrent = xArray[i,:]
        LCurrent = xCurrent[-1]
        w = int(wArray[i])
        # gCurrent = gArray[i,w]

        try:
            popt_cosh, pcov_cosh = curve_fit(FuncFit.cosh_func,xCurrent[w:N]/LCurrent,cCurrent[w:N],p0=[c_prefactor_sugg,phi_sugg],maxfev=1000000)
            c0_cosh = popt_cosh[0]
            phi = popt_cosh[1]
            lam_cosh_fit = phi*LCurrent
            decay_lengthArray = np.append(decay_lengthArray,abs(lam_cosh_fit))
            prefactorArray = np.append(prefactorArray,c0_cosh)
            pcov_decay_lengthArray = np.append(pcov_decay_lengthArray,pcov_cosh[1,1])
            pcov_prefactorArray = np.append(pcov_prefactorArray,pcov_cosh[0,0])

            # set new suggestions for phi and c_prefactor
            phi_sugg = phi
            c_prefactor_sugg = c0_cosh
        except (RuntimeWarning,OptimizeWarning):
            decay_lengthArray = np.append(decay_lengthArray,0)
            prefactorArray = np.append(prefactorArray,0)
            pcov_decay_lengthArray = np.append(pcov_decay_lengthArray,0)
            pcov_prefactorArray = np.append(pcov_prefactorArray,0)
            failed_fits = np.append(failed_fits,i)
        

    
    return prefactorArray, decay_lengthArray, failed_fits

def find_single_exp_fit_array(xArray,cArray,wArray,N,D,k): # find the decay length when fitting an exponentially decaying function to the morphogen profile
    decay_lengthArray = []
    prefactorArray = []
    pcovArray = []
    # suggested values for phi and the prefactor
    for i in range(len(xArray)):
        cCurrent = cArray[i,:]
        xCurrent = xArray[i,:]
        LCurrent = xCurrent[-1]
        w = int(wArray[i])
        # c0_sugg = cCurrent[w]
        # lam_sugg = np.sqrt(D/k)
        c0_sugg = cCurrent[w]
        # lam_sugg = np.sqrt(D/k)
        lam_sugg = LCurrent # for k=0
        popt, pcov = curve_fit(FuncFit.morph_dist,xCurrent[w:N],cCurrent[w:N],p0=[c0_sugg,lam_sugg])
        c0 = popt[0]
        lam = popt[1]
        decay_lengthArray = np.append(decay_lengthArray,abs(lam))
        prefactorArray = np.append(prefactorArray,c0)
        pcovArray = np.append(pcovArray,pcov)
        # c_prefactor_sugg = c0_cosh
        # phi_sugg = phi
    return prefactorArray, decay_lengthArray, pcovArray


# def find_decay_length_array_zero(xArray,cArray,wArray,N): # find the decay length when fitting an exponentially decaying function to the morphogen profile
#     decay_lengthArray = []
#     prefactorArray = []
#     pcovArray = []
#     # suggested values for phi and the prefactor
#     for i in range(len(xArray)):
#         cCurrent = cArray[i,:]
#         xCurrent = xArray[i,:]
#         LCurrent = xCurrent[-1]
#         w = int(wArray[i])
#         c0_sugg = cCurrent[w]
#         lam_sugg = LCurrent
#         popt, pcov = curve_fit(FuncFit.morph_dist,xCurrent[w:N],cCurrent[w:N],p0=[c0_sugg,lam_sugg])
#         c0 = popt[0]
#         lam = popt[1]
#         decay_lengthArray = np.append(decay_lengthArray,abs(lam))
#         prefactorArray = np.append(prefactorArray,c0)
#         pcovArray = np.append(pcovArray,pcov)
#     return prefactorArray, decay_lengthArray, pcovArray


def find_exp_x(xCurrent,cCurrent,wCurrent,LCurrent,sRatio,N):
    num_int = xCurrent*cCurrent
    return scipy.integrate.trapezoid(num_int[wCurrent:N],xCurrent[wCurrent:N])/scipy.integrate.trapezoid(cCurrent[wCurrent:N],xCurrent[wCurrent:N]) - LCurrent*sRatio

def find_exp_x_test(xCurrent,cCurrent,wCurrent,LCurrent,sRatio,N):
    x_shift = xCurrent - LCurrent*sRatio
    num_int = xCurrent*cCurrent
    return scipy.integrate.trapezoid(num_int[wCurrent:N],xCurrent[wCurrent:N])/scipy.integrate.trapezoid(cCurrent[wCurrent:N],xCurrent[wCurrent:N]) - LCurrent*sRatio


def find_exp_xArray(xArray,cArray,wArray,LArray,N,sRatio):
    N = int(N)
    exp_xArray = []
    for index in range(len(cArray)):
        cCurrent = cArray[index,:]
        xCurrent = xArray[index,:]
        wCurrent = int(wArray[index])
        num_int = xCurrent*cCurrent
        exp_x = scipy.integrate.trapezoid(num_int[wCurrent:N],xCurrent[wCurrent:N])/scipy.integrate.trapezoid(cCurrent[wCurrent:N],xCurrent[wCurrent:N]) - LArray[index]*sRatio
        exp_xArray = np.append(exp_xArray,exp_x)
    return exp_xArray




# Pattern quantification for genes
# def find_gene_boundary(geneArrayCurrent,xCurrent,fraction): # find the position where a gene reaches 80% of its maximum
#     # specify if function is increasing or decreasing
#     normGeneArray = geneArrayCurrent/max(geneArrayCurrent)
#     if geneArrayCurrent[0] < geneArrayCurrent[-1]: # an increasing function
#         index_m = np.searchsorted(normGeneArray,fraction)
#         gene_boundary = xCurrent[index_m] # the gene boundary without linear interpolation
#     elif geneArrayCurrent[0] > geneArrayCurrent[-1]: # a decreasing function
#         backwards_normGeneArray = np.flip(normGeneArray) # flip so that np.searchsorted can be used (only increasing functions)
#         backwards_x = np.flip(xCurrent)
#         index_m = np.searchsorted(backwards_normGeneArray,fraction)
#         gene_boundary = backwards_x[index_m]
#     return gene_boundary

# linear interpolation 
# removed source scaling capacity
def find_gene_boundary(geneArrayCurrent,xCurrent,fraction,linear_interpolation=True,norm_global=True,norm_value=1): # find the position where a gene reaches fraction% of its maximum
    # specify if function is increasing or decreasing
    # if ratio_given == True:
    #     source_size = LCurrent*sRatioOrSize
    # else:
    #     source_size = sRatioOrSize
    if norm_global == True: # normalise by the max of the whole distribution
        normGeneArray = geneArrayCurrent/norm_value
    else: # normalise by the current maximum
        normGeneArray = geneArrayCurrent/max(geneArrayCurrent)
    if geneArrayCurrent[0] < geneArrayCurrent[-1]: # an increasing function
        index_m = np.searchsorted(normGeneArray,fraction)
    
        if index_m == 0:
            return np.nan, 0
        if linear_interpolation == True:
            g_step = normGeneArray[index_m] - normGeneArray[index_m-1]
            x_step = xCurrent[index_m] - xCurrent[index_m-1]
            g_error = normGeneArray[index_m] - fraction 
            x_error = x_step*g_error/g_step
            # return xCurrent[index_m] - x_error - source_size, index_m # the distance of the boundary from the edge of the source
            return xCurrent[index_m] - x_error, index_m # the distance of the boundary from the edge of tissue
        
        else:
            # return xCurrent[index_m] - source_size, index_m 
            return xCurrent[index_m], index_m 
        
    else: # a decreasing function
        backwards_normGeneArray = np.flip(normGeneArray) # flip so that np.searchsorted can be used (only increasing functions)
        backwards_x = np.flip(xCurrent)
        index_b = np.searchsorted(backwards_normGeneArray,fraction)
        if index_b == 0:
            return np.nan, 0
        
        # backwards_x[index_b] - x_error - LCurrent*sRatio
        # index_m = len(geneArrayCurrent) - index_b - 1
        # reutrns boundary_x_value, boundary_index_value
        if linear_interpolation == True:
            g_step = backwards_normGeneArray[index_b] - backwards_normGeneArray[index_b-1] # linear interpolation
            x_step = backwards_x[index_b] - backwards_x[index_b-1]
            g_error = backwards_normGeneArray[index_b] - fraction 
            x_error = x_step*g_error/g_step # linearly interpolate the error in the gene boundary
            # return backwards_x[index_b] - x_error -source_size, len(geneArrayCurrent) - index_b - 1 # the distance of the boundary from the edge of the source
            return backwards_x[index_b] - x_error, len(geneArrayCurrent) - index_b - 1 # the distance of the boundary from left hand of tissue
        
        else:
            # return backwards_x[index_b] -source_size, len(geneArrayCurrent) - index_b - 1 # the distance of the boundary from the edge of the source
            return backwards_x[index_b], len(geneArrayCurrent) - index_b - 1 # geneboundary, geneboundaryindex
        
def find_fate_1_boundary(N1Current,xCurrent,MCurrent,V1,k1,h,Mstar,fraction=0.95):
    N1_fixedPointArray = (MCurrent**h/(Mstar**h + MCurrent**h))*V1/k1 # value of N1 at the fixed point
    # N2_fixedPointArray = FuncToggle.find_n2_SS_given_n1(N1_fixedPointArray,V2,k2,q,N1star)
    # find the % difference of each point in N1Current and N2Current to the fixed point
    N1_diff = np.abs(N1Current - N1_fixedPointArray)/N1_fixedPointArray
    # N2_diff = np.abs(N2Current - N2_fixedPointArray)/N2_fixedPointArray
    # find where the difference reaches 0.95
    index_m = np.searchsorted(N1_diff,fraction)
    # add in linear interpolation
    if index_m == 0:
        return 0
    elif index_m == len(N1Current):
        return xCurrent[-1]

    g_step = N1_diff[index_m] - N1_diff[index_m-1]
    x_step = xCurrent[index_m] - xCurrent[index_m-1]
    g_error = N1_diff[index_m] - fraction 
    x_error = x_step*g_error/g_step
    return xCurrent[index_m] - x_error

def find_fate_2_boundary(N2Current,xCurrent,V2,k2,fraction=0.95):
    N2_fixedPoint = V2/k2  # value of N2 at the fixed point
    # N1_fixedPoint = FuncToggle.find_n1_SS_given_n2(N2_fixedPoint,MCurrent,V1,k1,h,p,Mstar,N2star)
    # N1_diff = np.abs(N1Current - N1_fixedPoint)/N1_fixedPoint
    N2_diff = np.flip(np.abs(N2Current - N2_fixedPoint)/N2_fixedPoint)
    # find where the difference reaches 0.95
    index_m = np.searchsorted(N2_diff,fraction)
    xflip = np.flip(xCurrent)
    # add in linear interpolation
    if index_m == 0:
        print('index_m =0')
        return xCurrent[-1]
    elif index_m == len(N2Current):
        return 0

    g_step = N2_diff[index_m] - N2_diff[index_m-1]
    x_step = xflip[index_m] - xflip[index_m-1]
    g_error = N2_diff[index_m] - fraction 
    x_error = x_step*g_error/g_step
    return xflip[index_m] - x_error


@numba.njit(fastmath=True)
def find_fate_1_boundaryArray(N1Array:np.ndarray,xArray:np.ndarray,MArray:np.ndarray,V1,k1,h,Mstar,fraction=0.95):
    num_timepoints = len(N1Array)
    fate_1_boundaryArray = np.zeros(num_timepoints)
    for i in range(0,num_timepoints):
        N1Current = N1Array[i,:]
        xCurrent = xArray[i,:]
        N1_fixedPointArray = (MArray[i,:]**h/(Mstar**h + MArray[i,:]**h))*V1/k1 # value of N1 at the fixed point
        # N2_fixedPointArray = FuncToggle.find_n2_SS_given_n1(N1_fixedPointArray,V2,k2,q,N1star)
        # find the % difference of each point in N1Current and N2Current to the fixed point
        N1_diff = np.abs(N1Current - N1_fixedPointArray)/N1_fixedPointArray
        # N2_diff = np.abs(N2Current - N2_fixedPointArray)/N2_fixedPointArray
        # find where the difference reaches 0.95
        index_m = np.searchsorted(N1_diff,fraction)
        # add in linear interpolation
        if index_m == 0:
            fate_1_boundaryArray[i] = 0
        elif index_m == len(N1Current):
            fate_1_boundaryArray[i] = xCurrent[-1]
        else:
            g_step = N1_diff[index_m] - N1_diff[index_m-1]
            x_step = xCurrent[index_m] - xCurrent[index_m-1]
            g_error = N1_diff[index_m] - fraction 
            x_error = x_step*g_error/g_step
            fate_1_boundaryArray[i] = xCurrent[index_m] - x_error
    return fate_1_boundaryArray

@numba.njit(fastmath=True)
def find_fate_2_boundaryArray(N2Array:np.ndarray,xArray:np.ndarray,V2,k2,fraction=0.95):
    num_timepoints = len(N2Array)
    fate_2_boundaryArray = np.zeros(num_timepoints)
    N2_fixedPoint = V2/k2  # value of N2 at the fixed point
    for i in range(0,num_timepoints):
        N2Current = N2Array[i,:]
        xCurrent = xArray[i,:]
        # N1_fixedPoint = FuncToggle.find_n1_SS_given_n2(N2_fixedPoint,MCurrent,V1,k1,h,p,Mstar,N2star)
        # N1_diff = np.abs(N1Current - N1_fixedPoint)/N1_fixedPoint
        N2_diff = np.flip(np.abs(N2Current - N2_fixedPoint)/N2_fixedPoint)

        # find where the difference reaches 0.95
        index_m = np.searchsorted(N2_diff,fraction)
        xflip = np.flip(xCurrent)
        # add in linear interpolation
        if index_m == 0:
            fate_2_boundaryArray[i] = xCurrent[-1]
        elif index_m == len(N2Current):
            fate_2_boundaryArray[i] = 0
        else:
            g_step = N2_diff[index_m] - N2_diff[index_m-1]
            x_step = xflip[index_m] - xflip[index_m-1]
            g_error = N2_diff[index_m] - fraction 
            x_error = x_step*g_error/g_step
            fate_2_boundaryArray[i] = xflip[index_m] - x_error
    
    return fate_2_boundaryArray
# # removed LArray from inputs here - only used for scaling source
# def find_gene_boundaryArray_old(geneArray,xArray,sourceSize,fraction,scaledsource=False,linear_interpolation=True,init_boundary=0):
#     # assume that if the final state is an increasing/decreasing function, the whole time series is increasing/decreasing also
#     if init_boundary == 0:
#         gene_boundaryArray = [0] # initiate array to store all gene boundaries over time and set first value to 0
#         range_start = 1
#     else:
#         gene_boundaryArray = []
#         range_start = 0
#     # perform separately for increasing vs decreasing functions
#     if geneArray[-1,0] < geneArray[-1,-1]: # an increasing function. Assume it is either increasing or decreasing for the whole time
#         # iterate in time
#         for i in range(range_start,len(geneArray)):
#             # check if there is 0 concentration
#             if max(geneArray[i,:])==0: # if there is 0 concentration
#                 gene_boundary = 0 # set the position of the boundary to 0
#                 gene_boundaryArray = np.append(gene_boundaryArray,gene_boundary)

#             else: # continue if there is non-zero concentration
#                 # find the normalised gene array and the index where this is larger than fraction
#                 normGeneArray = geneArray[i,:]/max(geneArray[i,:])
#                 index_post = np.searchsorted(normGeneArray,fraction)

#                 # check this index is non-zero
#                 if index_post == 0: # if the gradient is too flat, fraction is never reached
#                     gene_boundary = np.nan
#                 # check this index is not the far edge of the tissue
#                 elif index_post == len(geneArray[i,:]):
#                     gene_boundary = np.nan

#                 else:
#                     xCurrent_post = xArray[i,index_post] # the x position just past the boundary
#                     xCurrent_pre = xArray[i,index_post-1] # the x position just before the boundary
#                     # find the size of the source - update according to L if it scales
#                     if scaledsource == True: # if the source is scaled, the size of the source is proportional to L, and a ratio has been given
#                         sourceSize = sourceSize*xArray[i,-1] # update the size of the source 
                    
#                     # perform linear interpolation, or not
#                     if linear_interpolation == True:
#                         g_step = normGeneArray[index_post] - normGeneArray[index_post-1]
#                         x_step = xCurrent_post - xCurrent_pre
#                         g_error = normGeneArray[index_post] - fraction 
#                         x_error = x_step*g_error/g_step
#                         # gene_boundary = xCurrent_post - x_error - sourceSize # the linearly interpolated gene boundary 
#                         gene_boundary = xCurrent_post - x_error # the linearly interpolated gene boundary 

#                     else: # without linear interpolation
#                         # gene_boundary = xCurrent_post - sourceSize # the NON lin interp gene boundary    
#                         gene_boundary = xCurrent_post # the NON lin interp gene boundary                    


            
#                 gene_boundaryArray = np.append(gene_boundaryArray,gene_boundary)

#     else: # a decreasing function
#         # iterate in time
#         for i in range(1,len(geneArray)):
#             # check if there is 0 concentration
#             if max(geneArray[i,:])==0: # if there is 0 concentration anywhere
#                 gene_boundary = 0 # set the position of the boundary to 0
#                 gene_boundaryArray = np.append(gene_boundaryArray,gene_boundary) 

#             else: # continue if there is non-zero concentration
#                 # find the normalised gene array and the index where this is larger than fraction
#                 normGeneArray = geneArray[i,:]/max(geneArray[i,:]) # find normalised gene array
#                 backwards_normGeneArray = np.flip(normGeneArray) # flip so that np.searchsorted can be used (which only works on increasing functions)
#                 backwards_x = np.flip(xArray[i,:]) # flip the x coordinates to ensure the values still correspond
#                 index_post = np.searchsorted(backwards_normGeneArray,fraction)
#                 print(index_post)
#                 # check this index is non-zero
#                 if index_post == 0: # if the gene conc never goes below fraction*max
#                     gene_boundary = np.nan
#                 # check this index is not the far edge of the tissue
#                 elif index_post == len(geneArray[i,:]):
#                     gene_boundary = np.nan
                              
#                 else:
#                     backwardsx_post = backwards_x[index_post] # the x position just past the boundary (in negative x space)
#                     backwards_x_pre = backwards_x[index_post-1] # the x position just before the boundary (in negative x space)
#                     # find the size of the source - update according to L if it scales
#                     if scaledsource == True: # if the source is scaled, the size of the source is proportional to L, and a ratio has been given
#                         sourceSize = sourceSize*xArray[i,-1] # update the size of the source 

#                     # perform linear interpolation, or not
#                     if linear_interpolation == True:
#                         g_step = backwards_normGeneArray[index_post] - backwards_normGeneArray[index_post-1]
#                         x_step = backwardsx_post - backwards_x_pre
#                         g_error = backwards_normGeneArray[index_post] - fraction 
#                         x_error = x_step*g_error/g_step
#                         # gene_boundary = backwardsx_post - x_error - sourceSize  # the linearly interpolated gene boundary 
#                         gene_boundary = backwardsx_post - x_error  # the linearly interpolated gene boundary 

#                     else: # without linear interpolation
#                         # gene_boundary = backwardsx_post - sourceSize
#                         gene_boundary = backwardsx_post

#                 gene_boundaryArray = np.append(gene_boundaryArray,gene_boundary)
    
#     return gene_boundaryArray

# removed LArray from inputs here - only used for scaling source
@numba.njit(fastmath=True)
def find_gene_boundaryArray(geneArray:np.ndarray,xArray:np.ndarray,sourceSizeOrRatio,fraction,scaledsource=False,linear_interpolation=True,norm_value=1,source_corrected=False):
    num_timepoints = len(geneArray)
    gene_boundaryArray = np.zeros(num_timepoints)
    # check whether the gradient is flat or not

    
    # assume that if the final state is an increasing/decreasing function, the whole time series is increasing/decreasing also
    # perform separately for increasing vs decreasing functions
    
    if geneArray[-1,0] < geneArray[-1,-1]: # an increasing function. Assume it is either increasing or decreasing for the whole time
        # iterate in time
        for i in range(0,num_timepoints):

            # check if there is 0 concentration
            if max(geneArray[i,:])==0: # if there is 0 concentration
                gene_boundaryArray[i] = 0 # set the position of the boundary to 0

            else: # continue if there is non-zero concentration
                # form normalised geneArray - global or non global
                normGeneArray = geneArray[i,:]/norm_value

                # find the index where the normalised gene array is larger than fraction
                index_post = np.searchsorted(normGeneArray,fraction)

                # check this index is non-zero
                if index_post == 0: # if the gradient is too flat, fraction is never reached
                    gene_boundary = np.nan
                # check this index is not the far edge of the tissue
                elif index_post == len(geneArray[i,:]):
                    gene_boundary = np.nan

                else:
                    xCurrent_post = xArray[i,index_post] # the x position just past the boundary
                    xCurrent_pre = xArray[i,index_post-1] # the x position just before the boundary
                    # find the size of the source - update according to L if it scales
                    if scaledsource == True and source_corrected == True: # if the source is scaled, the size of the source is proportional to L, and a ratio has been given
                        sourceSize_i = sourceSizeOrRatio*xArray[i,-1] # update the size of the source 
                    elif scaledsource == False and source_corrected == True:
                        sourceSize_i = sourceSizeOrRatio
                    else:
                        sourceSize_i = 0
                    
                    # perform linear interpolation, or not
                    if linear_interpolation == True:
                        g_step = normGeneArray[index_post] - normGeneArray[index_post-1]
                        x_step = xCurrent_post - xCurrent_pre
                        g_error = normGeneArray[index_post] - fraction 
                        x_error = x_step*g_error/g_step
                        # gene_boundary = xCurrent_post - x_error - sourceSize # the linearly interpolated gene boundary 
                        gene_boundary = xCurrent_post - x_error # the linearly interpolated gene boundary 

                    else: # without linear interpolation
                        # gene_boundary = xCurrent_post - sourceSize # the NON lin interp gene boundary    
                        gene_boundary = xCurrent_post # the NON lin interp gene boundary                    
            
                gene_boundaryArray[i] = gene_boundary - sourceSize_i

    else: # a decreasing function
        # iterate in time
        for i in range(0,len(geneArray)):

            # check if there is 0 concentration
            if max(geneArray[i,:])==0: # if there is 0 concentration anywhere
                gene_boundaryArray[i] = 0


            else: # continue if there is non-zero concentration
                normGeneArray = geneArray[i,:]/norm_value
                
                # find the normalised gene array and the index where this is larger than fraction
                backwards_normGeneArray = np.flip(normGeneArray) # flip so that np.searchsorted can be used (which only works on increasing functions)
                backwards_x = np.flip(xArray[i,:]) # flip the x coordinates to ensure the values still correspond
                index_post = np.searchsorted(backwards_normGeneArray,fraction)

                # check this index is non-zero
                if index_post == 0: # if the gene conc never goes below fraction*max
                    gene_boundary = np.nan
                elif index_post == len(geneArray[i,:]):
                    gene_boundary = np.nan
                              
                else:
                    backwardsx_post = backwards_x[index_post] # the x position just past the boundary (in negative x space)
                    backwards_x_pre = backwards_x[index_post-1] # the x position just before the boundary (in negative x space)
                    # find the size of the source - update according to L if it scales
                    if scaledsource == True and source_corrected == True: # if the source is scaled, the size of the source is proportional to L, and a ratio has been given
                        sourceSize_i = sourceSizeOrRatio*xArray[i,-1] # update the size of the source 
                    elif scaledsource == False and source_corrected == True:
                        sourceSize_i = sourceSizeOrRatio
                    else:
                        sourceSize_i = 0

                    # perform linear interpolation, or not
                    if linear_interpolation == True:
                        g_step = backwards_normGeneArray[index_post] - backwards_normGeneArray[index_post-1]
                        x_step = backwardsx_post - backwards_x_pre
                        g_error = backwards_normGeneArray[index_post] - fraction 
                        x_error = x_step*g_error/g_step
                        # gene_boundary = backwardsx_post - x_error - sourceSize  # the linearly interpolated gene boundary 
                        gene_boundary = backwardsx_post - x_error  # the linearly interpolated gene boundary 

                    else: # without linear interpolation
                        # gene_boundary = backwardsx_post - sourceSize
                        gene_boundary = backwardsx_post

                gene_boundaryArray[i] = gene_boundary - sourceSize_i
    
    return gene_boundaryArray

def find_gene_crossover(N1Current,N2Current,xCurrent,LCurrent,sRatio):
    norm_ratio = (N2Current/max(N2Current))/(N1Current/max(N1Current))
    index_cross = np.searchsorted(norm_ratio,1)
    g_step = norm_ratio[index_cross] - norm_ratio[index_cross-1]
    x_step = xCurrent[index_cross] - xCurrent[index_cross-1]
    g_error = norm_ratio[index_cross] - 1
    x_error = x_step*g_error/g_step
    # return xCurrent[index_cross] - x_error - sRatio*LCurrent
    return xCurrent[index_cross] - x_error



def find_gene_crossoverArray(N1Array,N2Array,xArray,LArray,sRatio,growing=1):
    no_timepoints = np.shape(N1Array)[0]
    no_spacepoints = np.shape(N1Array)[1]

    if growing == 0: # in the case where the system is not growing
        LArray = LArray*np.ones(no_timepoints) # make "LArray" which is 1 length into an array of the correct length
        xArray = np.reshape(list(xArray)*no_timepoints,(-1,no_spacepoints))

    gene_crossoverArray = [0] # initiate array to store all positions of gene crossover over time 
    gene_crossover_indices = [0]
    for i in range(1,no_timepoints): # interate for each time point
        if max(N2Array[i,:]==0) or max(N1Array[i,:]==0): # if the gene concentration is 0 everywhere
            gene_crossover = 0 # set the position of gene crossover to 0 
            gene_crossoverArray = np.append(gene_crossoverArray,gene_crossover)
        else:
            norm_ratio = (N2Array[i,:]/max(N2Array[i,:]))/(N1Array[i,:]/max(N1Array[i,:]))
            index_cross = np.searchsorted(norm_ratio,1)
            gene_crossover_indices = np.append(gene_crossover_indices,index_cross)
            xCurrent_post = xArray[i,index_cross]
            xCurrent_pre = xArray[i,index_cross-1]
            g_step = norm_ratio[index_cross] - norm_ratio[index_cross-1]
            x_step = xCurrent_post - xCurrent_pre
            g_error = norm_ratio[index_cross] - 1
            x_error = x_step*g_error/g_step
            gene_crossover = xArray[i,index_cross] - x_error - sRatio*LArray[i]
            gene_crossoverArray = np.append(gene_crossoverArray,gene_crossover)
    return gene_crossoverArray, gene_crossover_indices



    

def find_stat_point(RArray,timeArray):
    dt_store = timeArray[1]-timeArray[0]
    dRdt = np.gradient(RArray,dt_store)
    zero_indices = 2 + np.where(np.diff(np.sign(dRdt[2:])))[0] # finds indices where there is a sign change. dRdt is only valid from index 2 onwards, due to the curve fitting of lambda & the gradient function
    zero_index = zero_indices[0] # return the first occasion of reaching 0
    # linear interpolation to find estimate of time
    gap_dRdt = dRdt[zero_index+1] - dRdt[zero_index]
    zero_crossing_time = timeArray[zero_index] + dt_store*(-dRdt[zero_index])/gap_dRdt
    return zero_crossing_time, zero_indices



def find_crossing_point(timeArray,funcArray,val):
    shiftArray = funcArray - val
    zero_index = np.where(np.diff(np.sign(shiftArray)))[0]
    F_gap = shiftArray[zero_index+1] - shiftArray[zero_index] 
    t_gap = timeArray[zero_index+1] - timeArray[zero_index]
    F_error = -shiftArray[zero_index]
    t_zero = timeArray[zero_index] + t_gap*F_error/F_gap
    return t_zero

# different differentiating functions

def find_dFdt_manual(functionArray,dt_store): # find the difference in time for an array stored uniform every store_no of stores
    index_end = len(functionArray)
    dFdt = (functionArray[1:index_end] -  functionArray[0:index_end-1])/(dt_store)
    return dFdt


def find_dFdt_av(functionArray,dt_store,m): # find the difference in time for an array stored uniform every store_no of stores
    index_end = len(functionArray)
    dFdt = (functionArray[1:index_end] -  functionArray[0:index_end-1])/(dt_store)
    # average over every m values 
    group_size = int((index_end-1)/m)
    dFdt = dFdt[0:group_size*m]
    dFdtM = np.reshape(dFdt,(group_size,-1))
    dFdt = np.sum(dFdtM,1)/m
    return dFdt


def find_dFdt_points_linearinterp(functionArray,timeArray,m): # find the difference in time for an array stored in uniform time differences every store_no of stores
    index_end = len(functionArray)
    dFdt = np.gradient(functionArray,timeArray) # this uses adjacent points and thus is sufficient for the 2 outer indices on the left and right 
    half_m = int(m/2)

    for i in range(2,half_m):
        x = np.array(timeArray[0:i+1])
        y = np.array(functionArray[0:i+1])
        popt,pcov = curve_fit(FuncFit.straight_line,x,y)
        gradient = popt[0]
        dFdt[i] = gradient

    for i in range(half_m,index_end-half_m+1):
        x = np.array(timeArray[i-half_m:i+half_m+1])
        y = np.array(functionArray[i-half_m:i+half_m+1])
        popt,pcov = curve_fit(FuncFit.straight_line,x,y)
        gradient = popt[0]
        dFdt[i] = gradient
    
    for i in range(index_end-half_m,index_end-2):
        x = timeArray[i:index_end]
        y = functionArray[i:index_end]
        popt,pcov = curve_fit(FuncFit.straight_line,x,y)
        gradient = popt[0]
        dFdt[i] = gradient

    return dFdt

def find_cubicinterp(functionArray,timeArray,m): # find the difference in time for an array stored in uniform time differences every store_no of stores
    # m>4
    # this function automatically discounts the 1st point as it is always a poor fit for the decay length
    index_end = len(functionArray) # the length of the function
    F_fit = np.zeros(len(functionArray)) # initiate the fitted function and its 2 derivatives
    dFdt = np.zeros(len(functionArray))
    d2Fdt2 = np.zeros(len(functionArray))
    half_m = round(m/2) # the number of points either side. Total number of points used is m+1

    for i in range(1,half_m+1): # the points that have fewer than half_m to their left
        x = np.array(timeArray[1:i+half_m+1]) # points for fitting are bounded at 1
        y = np.array(functionArray[1:i+half_m+1])
        popt,pcov = curve_fit(FuncFit.cubic_line,x,y,maxfev=100000) # fit to a cubic
        F_fit[i] = FuncFit.cubic_line(timeArray[i],popt[0],popt[1],popt[2],popt[3]) # find the function for the fitted cubic
        dFdt[i] = 3*popt[0]*(timeArray[i]**2) + 2*popt[1]*timeArray[i] + popt[2]
        d2Fdt2[i] = 6*popt[0]*timeArray[i] + 2*popt[1]

    for i in range(half_m+1,index_end-half_m):
        x = np.array(timeArray[i-half_m:i+half_m+1])
        y = np.array(functionArray[i-half_m:i+half_m+1])
        popt,pcov = curve_fit(FuncFit.cubic_line,x,y,maxfev=20000)
        F_fit[i] = FuncFit.cubic_line(timeArray[i],popt[0],popt[1],popt[2],popt[3])
        dFdt[i] = 3*popt[0]*(timeArray[i]**2) + 2*popt[1]*timeArray[i] + popt[2]
        d2Fdt2[i] = 6*popt[0]*timeArray[i] + 2*popt[1]


    for i in range(index_end-half_m,index_end): # the points that have fewer than half_m to their right
        x = np.array(timeArray[i-half_m:index_end])
        y = np.array(functionArray[i-half_m:index_end])
        popt,pcov = curve_fit(FuncFit.cubic_line,x,y,maxfev=20000)
        F_fit[i] = FuncFit.cubic_line(timeArray[i],popt[0],popt[1],popt[2],popt[3])
        dFdt[i] = 3*popt[0]*(timeArray[i]**2) + 2*popt[1]*timeArray[i] + popt[2]
        d2Fdt2[i] = 6*popt[0]*timeArray[i] + 2*popt[1]

    return F_fit, dFdt, d2Fdt2




def find_max_deviation(sample,target):
    # abs_deviation = abs(sample - target)
    return max(abs(sample - target))

def find_max_deviation_sample(t_target,t_win,time_num,function): # function to find the variance of a function in a time window for a given start time
    # find the start and end times and their corresponding indices   
    t_end = t_target + t_win/2
    t_start = t_target - t_win/2
    # print(t_end,t_start)
    t_start_index = np.searchsorted(time_num,t_start)
    t_end_index = np.searchsorted(time_num,t_end)
    t_target_index = np.searchsorted(time_num,t_target)
    # linearly interpolate to find more accurate value of lp at t_target
    error_time = (time_num[t_target_index]-t_target)/(time_num[t_target_index]-time_num[t_target_index-1])
    error_func = error_time*(function[t_target_index]-function[t_target_index-1])

    sample = function[t_start_index:t_end_index+1]
    target = function[t_target_index] - error_func # as np.searchsorted always overshoots

    # t_win_real = time_num[t_end_index] - time_num[t_start_index] # the true value of t_win
    # num_sample = t_end_index - t_start_index + 1 # number of elements in the sample aka sample size
    # retrieve the sample from the function 
    
    sample_size = len(sample) 
    if sample_size<5:
        print("sample_size",sample_size)

    # find the max deviation from the target of this sample
    # abs_deviation = abs(sample - target)
    # max_dev = max(abs_deviation)
    return max(abs(sample - target))

def find_max_deviation_array(time_num,function,t_win,t_max,t_targetArray): # function to find the variances for a given time window at different start times
    max_devArray = []
    for i in range(len(t_targetArray)):
        t_target = t_targetArray[i]
        t_start = t_target - t_win/2
        t_end = t_start + t_win
        
        t_start_index = np.searchsorted(time_num,t_start)
        t_end_index = np.searchsorted(time_num,t_end)
        t_target_index = np.searchsorted(time_num,t_target)
        # linearly interpolate to find more accurate value of lp at t_target
        error_time = (time_num[t_target_index]-t_target)/(time_num[t_target_index]-time_num[t_target_index-1])
        error_func = error_time*(function[t_target_index]-function[t_target_index-1])

        sample = function[t_start_index:t_end_index+1]
        target = function[t_target_index] - error_func # as np.searchsorted always overshoots

        sample_size = len(sample)
        if sample_size<5:
            print("sample_size",sample_size)

        # print("t_start",t_start,"t_end",t_end,sample)
        # abs_deviation = abs(sample - target)
        max_dev = max(abs(sample - target))
        max_devArray = np.append(max_devArray,max_dev)
    return max_devArray

def find_norm_max_deviation_array(time_num,function,t_win,t_max,t_targetArray): # function to find the variances for a given time window at different start times
    max_devArray = []
    for i in range(len(t_targetArray)):
        t_target = t_targetArray[i]
        t_start = t_target - t_win/2
        t_end = t_start + t_win
        
        t_start_index = np.searchsorted(time_num,t_start)
        t_end_index = np.searchsorted(time_num,t_end)
        t_target_index = np.searchsorted(time_num,t_target)
        # linearly interpolate to find more accurate value of lp at t_target
        error_time = (time_num[t_target_index]-t_target)/(time_num[t_target_index]-time_num[t_target_index-1])
        error_func = error_time*(function[t_target_index]-function[t_target_index-1])

        sample = function[t_start_index:t_end_index+1]
        target = function[t_target_index] - error_func # as np.searchsorted always overshoots

        sample_size = len(sample)
        if sample_size<5:
            print("sample_size",sample_size)

        # print("t_start",t_start,"t_end",t_end,sample)
        # abs_deviation = abs(sample - target)
        max_dev = max(abs(sample - target))/target
        max_devArray = np.append(max_devArray,max_dev)
    return max_devArray

def find_norm_max_deviation_array_targets(time_num,function,t_win,t_max,t_targetArray): # function to find the variances for a given time window at different start times
    max_devArray = []
    f_targetArray = []
    for i in range(len(t_targetArray)):
        t_target = t_targetArray[i]
        t_start = t_target - t_win/2
        t_end = t_start + t_win
        
        t_start_index = np.searchsorted(time_num,t_start)
        t_end_index = np.searchsorted(time_num,t_end)
        t_target_index = np.searchsorted(time_num,t_target)
        # linearly interpolate to find more accurate value of function at t_target
        error_time = (time_num[t_target_index]-t_target)/(time_num[t_target_index]-time_num[t_target_index-1])
        error_func = error_time*(function[t_target_index]-function[t_target_index-1])

        sample = function[t_start_index:t_end_index+1]
        target = function[t_target_index] - error_func # as np.searchsorted always overshoots
        sample_size = len(sample)
        if sample_size<5:
            print("sample_size",sample_size)

        # print("t_start",t_start,"t_end",t_end,sample)
        # abs_deviation = abs(sample - target)
        max_dev = max(abs(sample - target))/target
        max_devArray = np.append(max_devArray,max_dev)
        f_targetArray = np.append(f_targetArray,target)
    return max_devArray, f_targetArray


def find_variance(sample):
    n = len(sample)
    mean_sample = sum(sample)/n
    deviations = [(x-mean_sample)**2 for x in sample]
    var_sample = sum(deviations)/n
    return var_sample

def find_variance_sample(t_target,t_win,time_num,function): # function to find the variance of a function in a time window for a given start time
    # find the start and end times, and their corresponding indices
    t_end = t_target + t_win/2
    t_start = t_target - t_win/2 
    t_start_index = np.searchsorted(time_num,t_start)
    t_end_index = np.searchsorted(time_num,t_end)
    # t_win_real = time_num[t_end_index] - time_num[t_start_index]
    # num_sample = t_end_index - t_start_index + 1
    # linearly interpolate to find more accurate value of lp at t_target
    sample = function[t_start_index:t_end_index+1]
    sample_size = len(sample)
    if sample_size<5:
        print("sample_size",sample_size)
    # find the variance of this sample
    var_sample = find_variance(sample) 
    return var_sample

def find_variance_array(time_num,function,t_win,t_targetArray): # function to find the variances for a given time window at different start times
    variances = []
    for i in range(len(t_targetArray)):
        t_target = t_targetArray[i]
        t_start = t_target - t_win/2
        t_end = t_start + t_win
        
        t_start_index = np.searchsorted(time_num,t_start)
        t_end_index = np.searchsorted(time_num,t_end)
        # t_win_real = time_num[t_end_index]-time_num[t_start_index]
        sample = function[t_start_index:t_end_index+1]
        variance = find_variance(function[t_start_index:t_end_index+1])
        variances = np.append(variances,variance)
    return variances

def find_variance_array_investigate(time_num,function,t_win,t_targetArray): # function to find the variances for a given time window at different start times
    variances = []
    sampleArray = []
    len_sampleArray = []
    for i in range(len(t_targetArray)):
        t_target = t_targetArray[i]
        t_start = t_target - t_win/2
        t_end = t_start + t_win
        
        t_start_index = np.searchsorted(time_num,t_start)
        t_end_index = np.searchsorted(time_num,t_end)
        # t_win_real = time_num[t_end_index]-time_num[t_start_index]
        sample = function[t_start_index:t_end_index+1]
        len_sampleArray = np.append(len_sampleArray,len(sample))
        sampleArray = np.append(sampleArray,sample)
        variance = find_variance(function[t_start_index:t_end_index+1])
        variances = np.append(variances,variance)
    return variances, sampleArray, len_sampleArray

def find_variance_R(t_target,t_win,time_num,x_ZArray,LArray): # find the variance in R from the variance in xthr and L
    t_end = t_target + t_win/2
    t_start = t_target - t_win/2 
    t_start_index = np.searchsorted(time_num,t_start)
    t_end_index = np.searchsorted(time_num,t_end)
    t_target_index = np.searchsorted(time_num,t_target)

    error_time = (time_num[t_target_index]-t_target)/(time_num[t_target_index]-time_num[t_target_index-1])
    error_L = error_time*(LArray[t_target_index]-LArray[t_target_index-1])
    error_xthr = error_time*(x_ZArray[t_target_index]-x_ZArray[t_target_index-1])

    target_L = LArray[t_target_index] - error_L
    target_xthr = x_ZArray[t_target_index] - error_xthr

    target_R = target_xthr/target_L

    sample_L = LArray[t_start_index:t_end_index+1]
    sample_xthr = LArray[t_start_index:t_end_index+1]

    variance_L = find_variance(LArray[t_start_index:t_end_index+1])
    variance_xthr = find_variance(x_ZArray[t_start_index:t_end_index+1])

    variance_R = (target_R**2)*((variance_L/(target_L**2))+(variance_xthr/(target_xthr**2)))
    return variance_R, target_R

def find_range(t_target,t_win,time_num,function): # function to find the range (difference in minimum and maximum values) of a function in a time window for a given start time
    t_start = t_target - t_win/2
    t_end = t_start + t_win
    
    t_start_index = np.searchsorted(time_num,t_start)
    t_end_index = np.searchsorted(time_num,t_end)
    range_f = np.ptp(function[t_start_index:t_end_index+1])
    return range_f

def find_range_array(time_num,function,t_win,t_targetArray):
    ranges = []
    for i in range(len(t_targetArray)):
        t_target = t_targetArray[i]
        t_start = t_target - t_win/2
        t_end = t_start + t_win
        
        t_start_index = np.searchsorted(time_num,t_start)
        t_end_index = np.searchsorted(time_num,t_end)
        # t_win_real = time_num[t_end_index]-time_num[t_start_index]
        range_f = np.ptp(function[t_start_index:t_end_index+1])
        ranges = np.append(ranges,range_f)
    return ranges


# function for finding the index of the source size
def find_w(LArray,sRatio,xArray,indexA):
    sSizeArray = LArray*sRatio
    return np.searchsorted(xArray[indexA,:],sSizeArray[indexA])
    # for i in range(len(xArray[indexA])):
    #     if xArray[indexA,i]>sSizeArray[indexA]:
    #         return i

def fit_stat_point_to_quad(timeArray,RArray,initial_time_offset=1): # if the initial conditions are not = 0, can set the initial_time_offset to 0 
    # function finds the stationary points of a function, then performs a local quadratic fit and outputs the quadratic parameters
    # find t_a 
    dt_store = timeArray[1]-timeArray[0]
    dRdt = np.gradient(RArray,dt_store)
    d2Rdt2 = np.gradient(dRdt,dt_store) # for use as estimate of a
    zero_indices = 1 + initial_time_offset + np.where(np.diff(np.sign(dRdt[2:])))[0] # finds indices where there is a sign change. dRdt is only valid from index 2 onwards, due to the curve fitting of lambda & the gradient function
    no_stat_points = len(zero_indices) 
    aArray = []
    t_aArray = []
    R_aArray = []
    perrArray = []
    for index in range(no_stat_points):
        zero_index = zero_indices[index] # return the first occasion of reaching 0
        # linear interpolation to find estimate of time
        gap_dRdt = dRdt[zero_index+1] - dRdt[zero_index]
        t_a = timeArray[zero_index] + dt_store*(-dRdt[zero_index])/gap_dRdt
        t_aArray = np.append(t_aArray,t_a)

        # fit the +- 5 points around t_a to a quadratic
        a_est = d2Rdt2[zero_index]
        R_a_est = RArray[zero_index]
        if zero_index < 5:
            lower_bound = 1
        else:
            lower_bound = zero_index - 4
        if zero_index > len(timeArray) + 4:
            upper_bound = len(timeArray)
        else:
            upper_bound = zero_index + 4
        popt, pcov = curve_fit(FuncFit.completed_square_quadratic_line,timeArray[lower_bound:upper_bound],RArray[lower_bound:upper_bound],p0=[a_est,t_a,R_a_est],maxfev=10000)
        aArray = np.append(aArray,popt[0]) # a = popt[0]
        R_aArray = np.append(R_aArray,popt[2]) # R_a = popt[2]
        perrArray = np.append(perrArray,np.sqrt(np.diag(pcov)))
    
    perrArray = np.reshape(perrArray,(-1,3))
    return aArray, t_aArray, R_aArray, perrArray # return the quadratic steepness/stepth/incline; the times of the stationary points, and the value of R at the stationary points, and errors in fits

# if multiple stationary points choose flattest at next step

def find_4_time_solutions(dRdt_val,dR2dt2_val,sigma):
    a = 0.5*dR2dt2_val
    b = dRdt_val
    c = sigma
    discriminant_p = b**2 + 4*a*c # for solving R + sigma
    discriminant_n = b**2 - 4*a*c # for solving R - sigma

    t_pp = (-b + np.sqrt(discriminant_p))/(2*a)
    t_pn = (-b - np.sqrt(discriminant_p))/(2*a)

    # solutions for R - sigma
    t_np = (-b + np.sqrt(discriminant_n))/(2*a)
    t_nn = (-b - np.sqrt(discriminant_n))/(2*a)
    return t_pp, t_pn, t_np, t_nn 

def find_time_windows_abs(RArray,timeArray,sigma,m=12): # find the time windows (negative and positive) that are within a precision of sigma in R
    # dRdt = np.gradient(RArray,dt_store)
    # d2Rdt2 = np.gradient(dRdt,dt_store)
    R_fit, dRdt, d2Rdt2 = find_cubicinterp(RArray,timeArray,m) 
    a = 0.5*d2Rdt2[3:]
    b = dRdt[3:]
    c = sigma
    N = len(timeArray)
    # initial arrays
    t_pp = np.zeros(N)
    t_pn = np.zeros(N)
    t_np = np.zeros(N)
    t_nn = np.zeros(N)
    # initiate minimum times
    t_n = np.zeros(N)
    t_p = np.zeros(N)

    discriminant_p = b**2 + 4*a*c # for solving R + sigma
    discriminant_n = b**2 - 4*a*c # for solving R - sigma

    # solutions for R + sigma
    t_pp[3:] = (-b + np.sqrt(discriminant_p))/(2*a)
    t_pn[3:] = (-b - np.sqrt(discriminant_p))/(2*a)

    # solutions for R - sigma
    t_np[3:] = (-b + np.sqrt(discriminant_n))/(2*a)
    t_nn[3:] = (-b - np.sqrt(discriminant_n))/(2*a)

    for i in range(N): # could be better way to do this than a loop
        t_values = [t_pp[i],t_pn[i],t_np[i],t_nn[i]]
        positive_values = [x for x in t_values if x > 0]
        negative_values = [x for x in t_values if x < 0]
        if len(positive_values)==0:
            t_p[i] = float("nan")
        else:
            t_p[i] = min(positive_values,key=abs)

        if len(negative_values)==0:
            t_n[i] = float("nan")
        else:
            t_n[i] = min(negative_values,key=abs)

    return [t_p,t_n]


def find_time_windows_rel(RArray,timeArray,sigma_factor,m=12): # find the time windows (negative and positive) that are within a precision of sigma in R
    R_fit, dRdt, d2Rdt2 = find_cubicinterp(RArray,timeArray,m) 
    a = 0.5*d2Rdt2[3:]
    b = dRdt[3:]
    c = sigma_factor*R_fit[3:]
    N = len(timeArray)
    # initial arrays
    t_pp = np.zeros(N)
    t_pn = np.zeros(N)
    t_np = np.zeros(N)
    t_nn = np.zeros(N)
    # initiate minimum times
    t_n = np.zeros(N)
    t_p = np.zeros(N)

    discriminant_p = b**2 + 4*a*c # for solving R + sigma
    discriminant_n = b**2 - 4*a*c # for solving R - sigma

    # solutions for R + sigma
    t_pp[3:] = (-b + np.sqrt(discriminant_p))/(2*a)
    t_pn[3:] = (-b - np.sqrt(discriminant_p))/(2*a)

    # solutions for R - sigma
    t_np[3:] = (-b + np.sqrt(discriminant_n))/(2*a)
    t_nn[3:] = (-b - np.sqrt(discriminant_n))/(2*a)

    for i in range(N): # could be better way to do this than a loop
        t_values = [t_pp[i],t_pn[i],t_np[i],t_nn[i]]
        positive_values = [x for x in t_values if x > 0]
        negative_values = [x for x in t_values if x < 0]
        if len(positive_values)==0:
            t_p[i] = float("nan")
        else:
            t_p[i] = min(positive_values,key=abs)

        if len(negative_values)==0:
            t_n[i] = float("nan")
        else:
            t_n[i] = min(negative_values,key=abs)

    return [t_p,t_n]

def find_time_window_num_point(RArray,timeArray,sigma,index,m_val):
    R_fit, dRdt, d2Rdt2 = find_cubicinterp(RArray,timeArray,m_val) 
    t_val = timeArray[index]
    R_val = R_fit[index]
    R_upper = R_val + sigma
    R_lower = R_val - sigma
    # find the closest values to R_upper
    t_zero_u = find_crossing_point(timeArray[1:],R_fit[1:],R_upper)
    t_zero_l = find_crossing_point(timeArray[1:],R_fit[1:],R_lower)
    windows_u = t_zero_u - t_val
    windows_l = t_zero_l - t_val
    t_values = np.append(windows_u,windows_l)
    positive_values = [x for x in t_values if x > 0]
    negative_values = [x for x in t_values if x < 0]

    if len(positive_values)==0:
        t_p = float("nan")
    else:
        t_p = min(positive_values,key=abs)

    if len(negative_values)==0:
        t_n = float("nan")
    else:
        t_n = min(negative_values,key=abs)
    
    return t_p, t_n

def find_time_windows_num(RArray_raw,timeArray,sigma,absolute=False,smooth=False,m_val=12): # timeArray must be [0,t_g]
    N = len(timeArray)
    RArray_raw[0] = 0
    if smooth == True:
        RArray, dRdt, d2Rdt2 = find_cubicinterp(RArray_raw,timeArray,m_val) 
    else:
        RArray = RArray_raw

    if absolute==True: # interpret sigma as an absolute tolerance
        sigmaArray = np.ones(N)*sigma # the tolerance is the same for each point
    else: # interpret sigma as the factor for a relative tolerance
        # sigmaArray = sigma*RArray # tolerance is proportional to R 
        sigmaArray = sigma*RArray # tolerance is proportional to R 


    t_n = np.zeros(N)
    t_p = np.zeros(N)
    for index in range(1,N):
        # R_val = RArray[index]
        R_val = RArray[index]

        t_val = timeArray[index]
        sigma_val = sigmaArray[index]
        R_upper = R_val + sigma_val
        R_lower = R_val - sigma_val

        # # find the closest values to R_upper
        t_zero_u = find_crossing_point(timeArray,RArray,R_upper)
        t_zero_l = find_crossing_point(timeArray,RArray,R_lower)


        # find the time window. find_crossing_point only gives the first crossing point but 
        windows_u = t_zero_u - t_val
        windows_l = t_zero_l - t_val
        t_values = np.append(windows_u,windows_l)

        positive_values = [x for x in t_values if x > 0]
        negative_values = [x for x in t_values if x < 0]

        if len(positive_values)==0:
            t_p[index] = float("nan")
        else:
            t_p[index] = min(positive_values,key=abs)
        if len(negative_values)==0:
            t_n[index] = float("nan")
        else:
            t_n[index] = min(negative_values,key=abs)

    nan_mask_n = np.isnan(t_n)
    t_n[nan_mask_n] = -timeArray[nan_mask_n]
    nan_mask_p = np.isnan(t_p)
    t_p[nan_mask_p] = timeArray[-1] - timeArray[nan_mask_p] # replace the nans with 
    return t_p,t_n


def find_norm_max_t_sigma(RArray,LArray,timeArray,m_val,sigma,g_max,alpha,epsilon,abs_bool=False,L_factor=0.98):
    # find t_g, the time it takes to reach L_factor% of final length
    L_f =  FuncAnalytical.find_Lf_exp_decay_growth(LArray[0],g_max,alpha,epsilon)
    t_g = FuncAnalytical.find_t_g_exp_decay_growth(LArray[0],g_max,alpha,epsilon,L_factor*L_f)

    # find numerical t_p and t_n
    [t_p,t_n] = find_time_windows_num(RArray,timeArray,sigma,absolute=abs_bool,smooth=False,m_val=m_val)
    max_sum_tau = max((t_p - (t_n)))
    norm_max_sum_tau = max_sum_tau/t_g
    return t_p, t_n, norm_max_sum_tau

def check_case(RArray,timeArray):
    
    # dRdt = np.gradient(RArray,sim_data.timeArray)
    m_val = max(int(len(timeArray)*0.1),6)
    try:
        R_smooth, dRdt_smooth, d2Rdt2_smooth = find_cubicinterp(RArray,timeArray,m_val)
        sign_array = np.sign(dRdt_smooth[:])
        sign_array[0] = 1 #  the first R value is always = 0, which gives 0 when np.sign acts on it. we're not interested in the change from 0 to positive.
        no_crossing_points = len(np.where(np.diff(sign_array))[0])
    except (RuntimeWarning,OptimizeWarning):
        no_crossing_points = -1

    if no_crossing_points==0 and R_smooth[2]<R_smooth[-1]:
        case = 1
    elif no_crossing_points==1 or (no_crossing_points==0 and max(R_smooth)>R_smooth[-1]):
        case = 2
    elif no_crossing_points==2:
        case = 3
    else:
        case = 4
    
    return case

def load_and_check_case(parent_dir,data_set_id):
    # load data
    dir = parent_dir + f"{data_set_id:04d}" + ".pkl"
    with open(dir, 'rb') as file:
        sim_data = pickle.load(file)
    D, k, sMag, sRatio, epsilon, g_max, alpha = sim_data.System_Parameters.values()
    Simulation_Parameters_values = sim_data.Simulation_Parameters.values()
    if len(Simulation_Parameters_values) == 6:
        N, dy, dt, timeSS, final_count, L_final = sim_data.Simulation_Parameters.values()
    else:
        N, dy, dt, timeSS, final_count, L_final, store_no = sim_data.Simulation_Parameters.values()

    # find decay length
    prefactorArray, decay_lengthArray, perrArray = find_decay_length_array(sim_data.xArray,sim_data.cArray,sim_data.wArray,N,sMag,D,k,sRatio)
    # R
    if len(sim_data.LArray)<20:
        with open("SmallDataSets2.txt", "a") as myfile:
            myfile.write(f"{data_set_id:04d}")

    RArray = decay_lengthArray/sim_data.LArray
    # dRdt = np.gradient(RArray,sim_data.timeArray)
    case = check_case(RArray,sim_data.timeArray)
    return case



def load_and_find_norm_max_t_sigma(data_set_id,parent_dir,m_val,sigma,L_factor=0.98):
    dir = parent_dir + f"{data_set_id:04d}" + ".pkl"
    with open(dir, 'rb') as file:
        sim_data = pickle.load(file)
    D, k, sMag, sRatio, epsilon, g_max, alpha = sim_data.System_Parameters.values()
    Simulation_Parameters_values = sim_data.Simulation_Parameters.values()
    if len(Simulation_Parameters_values) == 6:
        N, dy, dt, timeSS, final_count, L_final = sim_data.Simulation_Parameters.values()
    else:
        N, dy, dt, timeSS, final_count, L_final, store_no = sim_data.Simulation_Parameters.values()

    prefactorArray, decay_lengthArray, perrArray = find_decay_length_array(sim_data.xArray,sim_data.cArray,sim_data.wArray,N,sMag,D,k,sRatio)
    RArray = decay_lengthArray/sim_data.LArray
    L_f =  FuncAnalytical.find_Lf_exp_decay_growth(sim_data.LArray[0],g_max,alpha,epsilon)
    t_g = FuncAnalytical.find_t_g_exp_decay_growth(sim_data.LArray[0],g_max,alpha,epsilon,L_factor*L_f)

    # find numerical t_p and t_n
    [t_p_num,t_n_num] = find_time_windows_num(RArray,sim_data.timeArray,sigma,absolute=False,smooth=False,m_val=m_val)
    max_sum_tau = max((t_p_num - (t_n_num)))
    norm_max_sum_tau = max_sum_tau/t_g
    return norm_max_sum_tau


# %%%%%%%%%%%%%%%%%%%%%% Pattern quantification at the level of the GRN %%%%%%%%%%%%%%%%%%%%%%
# quantification of N1 and N2 profiles
# 0D fate quantification
def find_fate_at_SS(N1_value,N2_value,M_value,V1,k1,h,Mstar,V2,k2,precision=1E-2):
    n1_SS_theory_val = (V1/(k1))*(M_value**h/(Mstar**h + M_value**h)) # compute the theoretical maximum of N1 in fate 1
    n2_SS_theory_val = V2/k2 # compute the theoretical maximum of N2 in fate 2
    # check if N2 is at its maximium and N1 is at 0:
    # print(N1_value)
    # print(n1_SS_theory_val)
    # print(1- (n1_SS_theory_val - N1_value)/n1_SS_theory_val )
    # print(1- (n1_SS_theory_val - N1_value)/n1_SS_theory_val < 1E-2)
    if 1 - (n1_SS_theory_val - N1_value)/n1_SS_theory_val < precision and (n2_SS_theory_val - N2_value)/n2_SS_theory_val < precision:
        fate = 2
    # check if N2 is at its maximum and N1 is at 0
    elif 1 -(n2_SS_theory_val - N2_value)/n2_SS_theory_val < precision and (N1_value - N1_value)/N1_value < precision:
        fate = 1
    else:
        fate = np.nan
    return fate

# @numba.njit
# def find_fate_at_SSArray(n1_profile,n2_profile,m_profile,V1,k1,h,Mstar,V2,k2,precision=1E-2):
#     fateArray = np.zeros(len(n1_profile))
#     for space_index in range(len(n1_profile)): # iterate over space
#         M_value = m_profile[space_index]
#         N1_value = n1_profile[space_index]
#         N2_value = n2_profile[space_index]

#         n1_SS_theory_val = (V1/(k1))*(1/( (Mstar/M_value) **h )) # compute the theoretical maximum of N1 in fate 1
#         n2_SS_theory_val = V2/k2 # compute the theoretical maximum of N2 in fate 2
#         # check if N2 is at its maximium and N1 is at 0:
#         # print(N1_value)
#         # print(n1_SS_theory_val)
#         # print(1- (n1_SS_theory_val - N1_value)/n1_SS_theory_val )
#         # print(1- (n1_SS_theory_val - N1_value)/n1_SS_theory_val < 1E-2)
#         # check if n1_SS_theory_val is below a certain value
#         if n1_SS_theory_val < 1E-5:
#             n1_SS_theory_val = 1E-5

#         if 1- (n1_SS_theory_val - N1_value)/n1_SS_theory_val < precision and (n2_SS_theory_val - N2_value)/n2_SS_theory_val < precision:
#             fateArray[space_index] = 2
#         # check if N2 is at its maximum and N1 is at 0
#         elif 1-(n2_SS_theory_val - N2_value)/n2_SS_theory_val < precision and (N1_value - N1_value)/N1_value < precision:
#             fateArray[space_index] = 1
#         else:
#             fateArray[space_index] = np.nan
#     return 

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  dynamics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def find_index_plateau_indices_switch(N2_boundaryArray_1D_IC,timeArray_1D_IC,xArray_1D_IC,N2Array_1D_IC,x_crit_upper,g_max,fraction,V2,k2,grad_threshold_ratio=0.2,av_chunk_size=1):
    # average out the gradient of N2 boundary array
    grad_N2_boundaryArray_1D_IC = np.gradient(N2_boundaryArray_1D_IC)
    grad_threshold = grad_N2_boundaryArray_1D_IC[0] * grad_threshold_ratio
    grad_av = np.average(np.reshape(grad_N2_boundaryArray_1D_IC[:int(len(timeArray_1D_IC)/av_chunk_size)*av_chunk_size],(int(len(timeArray_1D_IC)/av_chunk_size),-1)),axis=1)
    if len(av_chunk_size*np.where(grad_av < grad_threshold)[0]) == 0:
        print('N2 boundary plateau not reached')
        return np.nan
    # find the plateau time
    index_plateau = av_chunk_size*np.where(grad_av < grad_threshold)[0][0]

    # find the index corresponding to the boundary at final time (i.e. the 'cell' that just becomes the N2 boundary)
    index_cell_N2_boundary_end = np.searchsorted(xArray_1D_IC[-1,:], N2_boundaryArray_1D_IC[-1])
    # find the index corresponding to the boundary at the start of the plateau (i.e. the final cell that just doesn't switch, it is N2 the whole time)
    index_cell_N2_boundary_plateau = np.searchsorted(xArray_1D_IC[index_plateau,:], N2_boundaryArray_1D_IC[index_plateau])
    return index_plateau, index_cell_N2_boundary_end, index_cell_N2_boundary_plateau

# using generated 1D simualtion data, find tau, the time it takes for a cell to become the N2 boundary after leaving the bistable region
#        average over the different spatial points that undergo this transition
# def quantify_tau_num_1D(N2_boundaryArray_1D_IC,timeArray_1D_IC,xArray_1D_IC,N2Array_1D_IC,x_crit_upper,g_max,fraction,V2,k2,grad_threshold_ratio=0.2,av_chunk_size=1):
#     # average out the gradient of N2 boundary array
#     grad_N2_boundaryArray_1D_IC = np.gradient(N2_boundaryArray_1D_IC)
#     grad_threshold = grad_N2_boundaryArray_1D_IC[0] * grad_threshold_ratio
#     grad_av = np.average(np.reshape(grad_N2_boundaryArray_1D_IC[:int(len(timeArray_1D_IC)/av_chunk_size)*av_chunk_size],(int(len(timeArray_1D_IC)/av_chunk_size),-1)),axis=1)
#     if len(av_chunk_size*np.where(grad_av < grad_threshold)[0]) == 0:
#         print('N2 boundary plateau not reached')
#         return np.nan
#     # find the plateau time
#     index_plateau = av_chunk_size*np.where(grad_av < grad_threshold)[0][0]

#     # find the index corresponding to the boundary at final time (i.e. the 'cell' that just becomes the N2 boundary)
#     index_cell_N2_boundary_end = np.searchsorted(xArray_1D_IC[-1,:], N2_boundaryArray_1D_IC[-1])
#     # find the index corresponding to the boundary at the start of the plateau (i.e. the final cell that just doesn't switch, it is N2 the whole time)
#     index_cell_N2_boundary_plateau = np.searchsorted(xArray_1D_IC[index_plateau,:], N2_boundaryArray_1D_IC[index_plateau])

#     tau_values = np.array([])
#     # iterate over trajectories for switching 'cells'
#     # for index in range(index_cell_N2_boundary_end+1,index_cell_N2_boundary_plateau):
#     #     index_leave_bistable = np.searchsorted(xArray_1D_IC[:,index],x_crit_upper) # find the index when the cell leaves the bistable region using the critical value
#     #     index_N2_enter =  np.searchsorted(N2Array_1D_IC[:,index],fraction*V2/(k2+g_max)) # find the index when the cell hits the N2 threshold (for this growth)
#     #     tau = timeArray_1D_IC[index_N2_enter] - timeArray_1D_IC[index_leave_bistable] # find the time difference 
#     #     tau_values = np.append(tau_values, tau)

#     index_edge_bistable = np.searchsorted(xArray_1D_IC[0,:], x_crit_upper)
#     for index in range(index_cell_N2_boundary_end+1,index_edge_bistable):
#         xArray_at_index = xArray_1D_IC[:,index]
#         index_leave_bistable = np.searchsorted(xArray_at_index,x_crit_upper) # find the index when the cell leaves the bistable region using the critical value
#         # linear interpolation to correct error
#         x_error = xArray_at_index[index_leave_bistable] - x_crit_upper 
#         x_step = xArray_at_index[index_leave_bistable] - xArray_at_index[index_leave_bistable-1]
#         t_step_bi = timeArray_1D_IC[index_leave_bistable] - timeArray_1D_IC[index_leave_bistable-1]
#         t_error_bi = (x_error/x_step) * t_step_bi
#         time_leave_bistable = timeArray_1D_IC[index_leave_bistable] - t_error_bi 
#         N2Array_at_index = N2Array_1D_IC[:,index]
#         # assumes monotonic n2
#         if N2Array_at_index[0] > fraction*V2/(k2+g_max):
#             tau = 0
#         else:
#             index_N2_enter =  np.searchsorted(N2Array_at_index,fraction*V2/(k2+g_max)) # find the index when the cell hits the N2 threshold (for this growth)
#             if index_N2_enter == len(N2Array_at_index):
#                 tau = np.nan
#             else: 
#                 # linear interpolation
#                 N2_error = N2Array_at_index[index_N2_enter] - fraction*V2/(k2+g_max) 
#                 N2_step = N2Array_at_index[index_N2_enter] - N2Array_at_index[index_N2_enter-1]
#                 t_step = timeArray_1D_IC[index_N2_enter] - timeArray_1D_IC[index_N2_enter-1]
#                 t_error = (N2_error/N2_step) * t_step
#                 time_N2_boundary = timeArray_1D_IC[index_N2_enter] - t_error
#                 tau = time_N2_boundary - time_leave_bistable # find the time difference 
#                 print(xArray_at_index[0],time_N2_boundary, time_leave_bistable, tau)
#         tau_values = np.append(tau_values, tau)

#     return np.average(tau_values)

# a function to find tau (the respecification time for a cell to move from the edge of the bistable region to the N2 boundary from the 1D data) )
def quantify_tau_1D(N2_boundaryArray_1D_IC,timeArray_1D_IC,xArray_1D_IC,MArray_1D_IC,N2Array_1D_IC,M_critical1,g_max,fraction,V2,k2):
    # works for monotonic or non-monotonic MArray_at_index

    # find the index for the cell which starts at Mcritical, the edge of the bistable region (and thus immediately leaves)
    index_edge_bistable = len(MArray_1D_IC[0,:]) - np.searchsorted(np.flip(MArray_1D_IC[0,:]), M_critical1)

    # find the index corresponding to the boundary at final time (i.e. the cell that just becomes the N2 boundary)
    index_cell_N2_boundary_end = np.searchsorted(xArray_1D_IC[-1,:], N2_boundaryArray_1D_IC[-1])

    indices = np.arange(index_cell_N2_boundary_end+1,index_edge_bistable,dtype=int) # all the cells that leave the bistable region and go on to become the N2 boundary
    indices_tau = np.array([],dtype=int) # initiate an array to store indices where we can find tau
    tau_values = np.array([])
    time_N2_boundary_values = np.array([])
    # iterate over trajectories for different initial positions
    for index in indices: # iterate through possible indices
        # use MArray and Mcritical to find when the cell leaves the bistable region
        MArray_at_index = MArray_1D_IC[:,index] # slice to find M tilde for this initial position
        # find where marray = Mcritical1 for general non-monotonic array
        Mcrit_crossings = np.where(np.diff(np.sign(MArray_at_index - M_critical1))) # find indices just before crossings 
        if len(Mcrit_crossings[0]) != 1:  # if there are 0 or multiple crossings
            print(f'No unique crossing of Mcritical for initial position index {index}')
            continue # skip to next index

        # find the time when the cell leaves the bistable region
        index_leave_bistable = Mcrit_crossings[0][0]
        M_error = MArray_at_index[index_leave_bistable] - M_critical1
        M_step = MArray_at_index[index_leave_bistable] - MArray_at_index[index_leave_bistable-1]
        t_step_bi = timeArray_1D_IC[index_leave_bistable] - timeArray_1D_IC[index_leave_bistable-1]
        t_error_bi = (M_error/M_step) * t_step_bi
        time_leave_bistable = timeArray_1D_IC[index_leave_bistable] - t_error_bi 


        N2Array_at_index = N2Array_1D_IC[:,index]
        # assumes monotonic n2
        if N2Array_at_index[0] > fraction*V2/(k2+g_max): # cells that were already the N2 boundary 
            # print(f'Cell was already at or above N2 boundary threshold for initial position index {index}')
            continue # skip to next index
        
        # assumes that N2Array_at_index is monotonically increasing
        index_N2_enter =  np.searchsorted(N2Array_at_index,fraction*V2/(k2+g_max)) # find the index when the cell hits the N2 threshold (for this growth)
        if index_N2_enter == len(N2Array_at_index): # if the index where the cell hits the N2 threshold is final / beyond
            print(f'Never reaches N2 boundary for initial position index {index}')
            continue # skip to next index

        # linear interpolation
        N2_error = N2Array_at_index[index_N2_enter] - fraction*V2/(k2+g_max) 
        N2_step = N2Array_at_index[index_N2_enter] - N2Array_at_index[index_N2_enter-1]
        t_step = timeArray_1D_IC[index_N2_enter] - timeArray_1D_IC[index_N2_enter-1]
        t_error = (N2_error/N2_step) * t_step
        time_N2_boundary = timeArray_1D_IC[index_N2_enter] - t_error
        tau = time_N2_boundary - time_leave_bistable # find the time difference 
        # print(MArray_at_index[0],time_N2_boundary, time_leave_bistable, tau)
        tau_values = np.append(tau_values, tau)
        time_N2_boundary_values = np.append(time_N2_boundary_values, time_N2_boundary)
        indices_tau = np.append(indices_tau, index)

    return tau_values, indices_tau, time_N2_boundary_values

# using generated 0D simulations, find tau, the time it takes for a cell to become the N2 boundary after leaving the bistable region
def quantify_tau_num_0D(timeArray_0D,N2Array_0D,xArray_0D,x_crit_upper,fraction,V2,k2,g_max,index_leave_bistable=1):
    if index_leave_bistable == 1:
        index_leave_bistable = np.searchsorted(xArray_0D,x_crit_upper) # find the index when the cell leaves the bistable region using the critical value
    index_N2_enter =  np.searchsorted(N2Array_0D,fraction*V2/(k2+g_max)) # find the index when the cell hits the N2 threshold (for this growth)
    # linear interpolation
    N2_error = N2Array_0D[index_N2_enter] - fraction*V2/(k2+g_max) 
    N2_step = N2Array_0D[index_N2_enter] - N2Array_0D[index_N2_enter-1]
    t_step = timeArray_0D[index_N2_enter] - timeArray_0D[index_N2_enter-1]
    t_error = (N2_error/N2_step) * t_step
    tau = timeArray_0D[index_N2_enter] - t_error - timeArray_0D[index_leave_bistable] # find the time difference 
    return tau

def quantify_tau_from_M_tilde(M_tilde,M_critical1,g0,V1,V2,h,p,q,k1,k2,Mstar,N1star,N2star,fraction,dt_0D,
                                        t_max_factor=7.5):
    """ A function to quantify tau for a given inputted M tilde (the morphogen signal received).
        Returns the tau value along with the full N1, N2, time arrays from the 0D simulation.
        Returns the time the cell reaches the N2 boundary and the time it leaves the bistable region for use calculating Delta.
    """
    t_max_ng = t_max_factor*(1/k1 + 1/k2)
    count_max_ng = round(t_max_ng/dt_0D) # the number of time points that will be computed - round this up
    
    # generate initial conditions by simulating to steady state for M = M_tilde[0]
    N1Array_for_IC, N2Array_for_IC, timeArray_for_IC, SS_check_N1_for_IC, SS_check_N2_for_IC = FuncToggle.N1N2_0D_Growth_solver(M_tilde[0] * np.ones(count_max_ng),g0*np.ones(count_max_ng),V1,V2,h,p,q,k1,k2,Mstar,N1star,N2star,dt_0D,V1/(k1+g0),0)
    
    # solve the 0D system with M tilde as the input to find the N1 and N2 solutions in time
    N1Array_0D_i, N2Array_0D_i, timeArray_0D_i, SS_check_N1_0D_i, SS_check_N2_0D_i = FuncToggle.N1N2_0D_Growth_solver(M_tilde,g0*np.ones(len(M_tilde)),V1,V2,h,p,q,k1,k2,Mstar,N1star,N2star,dt_0D,N1_init=N1Array_for_IC[-1],N2_init=0)

    # find when N2 = fraction*V2/(k2+g_max)
    N2boundary_crossings = np.where(np.diff(np.sign(N2Array_0D_i - fraction*V2/(k2+g0)))) # find indices just before crossings 
    if len(N2boundary_crossings[0]) != 1:  # if there are 0 or multiple crossings
        print(f'No unique crossing of N2boundary_crossings')
        return
    # find the time when the cell leaves the bistable region
    index_N2_enter = N2boundary_crossings[0][0]
    # index_N2_enter =  np.searchsorted(N2Array_0D_i,fraction*V2/(k2+g0)) # find the index when the cell reaches the N2 threshold (for this growth)
    if index_N2_enter == len(N2Array_0D_i): # if the 
        tau = np.nan
        time_leave_bistable = np.nan
        time_N2_boundary = np.nan
        print('Never reaches N2 boundary')
    else: 
        # find when M tilde leaves the bistable region
        Mcrit_crossings = np.where(np.diff(np.sign(M_tilde - M_critical1))) # find indices just before crossings 
        if len(Mcrit_crossings[0]) != 1:  # if there are 0 or multiple crossings
            print(f'No unique crossing of Mcritical')
            return

        # find the time when the cell leaves the bistable region
        index_leave_bistable = Mcrit_crossings[0][0]
        # linear interpolation to correct error
        M_error = M_tilde[index_leave_bistable] - M_critical1 
        M_step = M_tilde[index_leave_bistable] - M_tilde[index_leave_bistable-1]
        t_step_bi = timeArray_0D_i[index_leave_bistable] - timeArray_0D_i[index_leave_bistable-1]
        t_error_bi = (M_error/M_step) * t_step_bi
        time_leave_bistable = timeArray_0D_i[index_leave_bistable] - t_error_bi 
        # linearly interpolate to correct error
        N2_error = N2Array_0D_i[index_N2_enter] - fraction*V2/(k2+g0) 
        N2_step = N2Array_0D_i[index_N2_enter] - N2Array_0D_i[index_N2_enter-1]
        t_step = timeArray_0D_i[index_N2_enter] - timeArray_0D_i[index_N2_enter-1]
        t_error = (N2_error/N2_step) * t_step
        time_N2_boundary = timeArray_0D_i[index_N2_enter] - t_error
        tau = time_N2_boundary - time_leave_bistable # find the time difference
    # tauArray[i], n2_leave_bistable, n1_leave_bistable, N1Array_0D_i, N2Array_0D_i, timeArray_0D_i
    # return tau, N2Array_0D_i[index_leave_bistable], N1Array_0D_i[index_leave_bistable], N1Array_0D_i, N2Array_0D_i, timeArray_0D_i, time_N2_boundary, time_leave_bistable
    return tau, N1Array_0D_i, N2Array_0D_i, timeArray_0D_i, time_N2_boundary, time_leave_bistable

# for a given GRN, morphogen profile and growth, simulate the M tilde a cell will feel (in the limit of the morphogen profile being static) 
# and find tau,  the time it takes for a cell to become the N2 boundary after leaving the bistable region
def find_tau_num_0D(V1,V2,h,p,q,k1,k2,Mstar,N1star,N2star,vM,k,D,g0,epsilon,L0,sourceSize,x_critical1,fraction,store_no_0D):
    x_initial = sourceSize # start from edge of the source, assuming this is < the edge of the bistable region
    t_g = FuncAnalytical.find_t_g_constant_growth(x_initial,g0,epsilon,x_critical1*10)
    dt_0D = min(t_g/store_no_0D,0.1/(k1+g0))*0.1
    timeArray_0D_i = np.arange(0,t_g,dt_0D)
    # generate M tilde
    M_tilde_i = FuncToggle.find_M_tilde_constant_growth(L0,g0,epsilon,sourceSize,vM,k,D,x_initial,timeArray_0D_i)
    # n1_init0D_full,n2_init0D_full = V1/(k1), 0
    n1_init0D_full,n2_init0D_full = V1/(k1+g0), 0 # testing?

    # 0D simulation
    N1Array_0D_i, N2Array_0D_i, timeArray_0D_i, SS_check_N1_0D_i, SS_check_N2_0D_i = FuncToggle.N1N2_0D_Growth_solver(M_tilde_i,g0*np.ones(len(M_tilde_i)),V1,V2,h,p,q,k1,k2,Mstar,N1star,N2star,dt_0D,n1_init0D_full,n2_init0D_full)
    xArray_0D_i = FuncAnalytical.find_L_ana_constant_growth(x_initial,g0,epsilon,timeArray_0D_i)
    tau = quantify_tau_num_0D(timeArray_0D_i,N2Array_0D_i,xArray_0D_i,x_critical1,fraction,V2,k2,g0,index_leave_bistable=1)
    return tau

# Delta is the transient scaling distance: calculated by finding the difference between the N2 boundary value when it plateaus and the edge of the bistable region
def find_Delta_num(N2_boundaryArray_1D_IC,timeArray_1D_IC,x_crit_upper,grad_threshold_ratio=0.2,av_chunk_size=5):
    grad_N2_boundaryArray_1D_IC = np.gradient(N2_boundaryArray_1D_IC)
    if np.isnan(grad_N2_boundaryArray_1D_IC[0]):
        grad_threshold = abs(grad_N2_boundaryArray_1D_IC[~np.isnan(grad_N2_boundaryArray_1D_IC)][0]) * grad_threshold_ratio
    else:
        grad_threshold = abs(grad_N2_boundaryArray_1D_IC[0]) * grad_threshold_ratio
    grad_av = np.average(np.reshape(grad_N2_boundaryArray_1D_IC[:int(len(timeArray_1D_IC)/av_chunk_size)*av_chunk_size],(int(len(timeArray_1D_IC)/av_chunk_size),-1)),axis=1)
    # check if grad_av goes below the threshold; i.e. if the N2 boundary plateaus
    # making the grad_threshold smaller will exert a more stringent condition to detect the plateau.
    if len(av_chunk_size*np.where(grad_av < grad_threshold)[0]) == 0:
        print('Ni boundary plateau not reached: Delta not found')
        return np.nan
    index_plateau = av_chunk_size*np.where(grad_av < grad_threshold)[0][0]
    return np.average(N2_boundaryArray_1D_IC[int(index_plateau)+1:]) - x_crit_upper

# find max scaling length numerically - the tissue length value of index_plateau
def find_max_L_scaling_num(N2_boundaryArray_1D_IC,timeArray_1D_IC,LArray_1D_IC,grad_threshold_ratio=0.2,av_chunk_size=1):
    # find index_plateau from N2 boundary array 
    # average out the gradient of N2 boundary array
    grad_N2_boundaryArray_1D_IC = np.gradient(N2_boundaryArray_1D_IC)
    grad_threshold = grad_N2_boundaryArray_1D_IC[0] * grad_threshold_ratio
    grad_av = np.average(np.reshape(grad_N2_boundaryArray_1D_IC[:int(len(timeArray_1D_IC)/av_chunk_size)*av_chunk_size],(int(len(timeArray_1D_IC)/av_chunk_size),-1)),axis=1)
    if len(av_chunk_size*np.where(grad_av < grad_threshold)[0]) == 0:
        print('Ni boundary plateau not reached')
        return np.nan
    # find the plateau time
    index_plateau = av_chunk_size*np.where(grad_av < grad_threshold)[0][0]

    # find the L value at index_plateau
    return LArray_1D_IC[index_plateau]

def find_max_Lchi_scaling_num(N2_boundaryArray_1D_IC,timeArray_1D_IC,LArray_1D_IC,grad_threshold_ratio=0.2,av_chunk_size=1):
    # find index_plateau from N2 boundary array 
    # average out the gradient of N2 boundary array
    grad_N2_boundaryArray_1D_IC = np.gradient(N2_boundaryArray_1D_IC)
    grad_threshold = grad_N2_boundaryArray_1D_IC[0] * grad_threshold_ratio
    grad_av = np.average(np.reshape(grad_N2_boundaryArray_1D_IC[:int(len(timeArray_1D_IC)/av_chunk_size)*av_chunk_size],(int(len(timeArray_1D_IC)/av_chunk_size),-1)),axis=1)
    if len(av_chunk_size*np.where(grad_av < grad_threshold)[0]) == 0:
        print('N2 boundary plateau not reached')
        return np.nan
    # find the plateau time
    index_plateau = av_chunk_size*np.where(grad_av < grad_threshold)[0][0]

    # find the L value at index_plateau
    return LArray_1D_IC[index_plateau], N2_boundaryArray_1D_IC[index_plateau]

def find_xcrit_at_index(cell_indices,M_critical1,MArray_1D,xArray_1D,timeArray_1D):
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
        MArray_at_index = MArray_1D[:,cell_index] # slice to find M tilde at this index
        # find when this cell crosses Mcritical
        Mcrit_crossings = np.where(np.diff(np.sign(MArray_at_index - M_critical1))) # find indices just before crossings 

        index_leave_bistable_i = Mcrit_crossings[0][0] # the index in time where this cell leaves the bistable region
        # linear interpolation to find error
        M_error = MArray_at_index[index_leave_bistable_i] - M_critical1
        M_step = MArray_at_index[index_leave_bistable_i] - MArray_at_index[index_leave_bistable_i-1]
        t_step_bi = timeArray_1D[index_leave_bistable_i] - timeArray_1D[index_leave_bistable_i-1]
        t_error_bi = (M_error/M_step) * t_step_bi
        time_leave_bistableArray[i] = timeArray_1D[index_leave_bistable_i] - t_error_bi

        # find xcritical from the m profile at the time that this cell leaves the bistable region
        xcritArray[i] = FuncToggle.find_x_critical_from_M_critical(MArray_1D[index_leave_bistable_i,:],M_critical1, xArray_1D[index_leave_bistable_i,:],len(xArray_1D[index_leave_bistable_i,:]))

    return xcritArray, time_leave_bistableArray

def find_chi_ana_from_tau(g0, epsilon, L0, chi_0, x_crit_values, tau_values, time_N2_boundary_values_1D, num_points=100):
    chi_star = x_crit_values * ( np.exp( (g0 * tau_values) / (1 + epsilon)))
    t_star = FuncAnalytical.find_t_g_constant_growth(chi_0,g0,epsilon,chi_star[-1]) # find how long it takes to go from chi_0 to chi_star
    t_scale = np.linspace(0,t_star,num_points)
    chi_scale = FuncAnalytical.find_L_ana_constant_growth(chi_0, g0 ,epsilon, t_scale)
    L_scale = FuncAnalytical.find_L_ana_constant_growth(L0, g0 ,epsilon, t_scale)
    L_star = FuncAnalytical.find_L_ana_constant_growth(L0,g0,epsilon,time_N2_boundary_values_1D)
    return np.append(chi_scale,np.flip(chi_star)), np.append(L_scale,np.flip(L_star))

# with source growth
def find_chi_ana_from_tau_sg(g0, epsilon, L0, chi_0, xcritArray, tau_values, time_N2_boundary_values, num_points=100):
    chi_star_values = xcritArray * ( np.exp( (g0 * tau_values) / (1 + epsilon)))
    L_star_values = FuncAnalytical.find_L_ana_constant_growth(L0,g0,epsilon,time_N2_boundary_values)
    t_scale = np.linspace(0,time_N2_boundary_values[-1],num_points)
    chi_scale = FuncAnalytical.find_L_ana_constant_growth(chi_0, g0 ,epsilon, t_scale)
    L_scale = FuncAnalytical.find_L_ana_constant_growth(L0, g0 ,epsilon, t_scale)
    return np.append(chi_scale,np.flip(chi_star_values)), np.append(L_scale,np.flip(L_star_values)), np.append(t_scale,np.flip(time_N2_boundary_values))

def find_chi_ana_quantify_tau_1D(M_critical1,g_max,epsilon,L0,fraction,V2,k2,timeArray_1D_IC,xArray_1D_IC,MArray_1D_IC,N2Array_1D_IC,N2_boundaryArray_1D_IC):
    tauArray_1D, indices_1D, time_N2_boundary_values_1D = quantify_tau_1D(N2_boundaryArray_1D_IC,timeArray_1D_IC,xArray_1D_IC,MArray_1D_IC,N2Array_1D_IC,M_critical1,g_max,fraction,V2,k2)
    xcrit_from_xArray_func, time_leave_bistableArray_func = find_xcrit_at_index(indices_1D,M_critical1,MArray_1D_IC,xArray_1D_IC,timeArray_1D_IC)
    chi_0 = N2_boundaryArray_1D_IC[0]
    chi_ana, L_values_chi = find_chi_ana_from_tau(g_max, epsilon, L0, chi_0, xcrit_from_xArray_func, tauArray_1D, time_N2_boundary_values_1D)
    return chi_ana, L_values_chi

def find_chi_ana_quantify_tau_1D_sg( M_critical1, g0, epsilon, L0, fraction, V2, k2, timeArray_1D, xArray_1D, MArray_1D, N2Array_1D_IC, N2_boundaryArray_1D_IC, num_points=100):
    # find tau and corresponding time of becoming N2 boundary from 1D arrays
    tau_values, indices_1D_sg, time_N2_boundary_values = quantify_tau_1D(N2_boundaryArray_1D_IC,timeArray_1D,xArray_1D,MArray_1D,N2Array_1D_IC,M_critical1,g0,fraction,V2,k2)
    # find the values of x critical at the time of leaving the bistable region from the 1D arrays
    xcritArray, time_leave_bistableArray = find_xcrit_at_index(indices_1D_sg,M_critical1,MArray_1D,xArray_1D,timeArray_1D)
    chi_0 = N2_boundaryArray_1D_IC[0]
    chi_ana, L_values_chi, time_values_chi = find_chi_ana_from_tau_sg(g0, epsilon, L0, chi_0, xcritArray, tau_values, time_N2_boundary_values)
    return chi_ana, L_values_chi, time_values_chi

# static xcritical1 - the maximum boundary position (chi_star) is static too
def find_chi_star_ana_0D_tau_from_xcrit(x_critical1,epsilon,L0,sourceSize0,fraction,V1,V2,h,p,q,k1,k2,Mstar,N1star,N2star,vM,k,D,g0,store_no_0D=500):
    # first generate tau from the 0D simulation
    tau_0D = find_tau_num_0D(V1,V2,h,p,q,k1,k2,Mstar,N1star,N2star,vM,k,D,g0,epsilon,L0,sourceSize0,x_critical1,fraction,store_no_0D)
    chi_star = x_critical1 * ( np.exp( (g0 * tau_0D) / (1 + epsilon)))
    return chi_star

# source growth

# def find_Delta_ana_from_tau_sg(g0, epsilon, L0, xcritArray, tau_values, time_N2_boundary_values):
#     Delta_values = xcritArray * ( np.exp( (g0 * tau_values) / (1 + epsilon)) - 1)
#     L_star_values = FuncAnalytical.find_L_ana_constant_growth(L0,g0,epsilon,time_N2_boundary_values)
#     return Delta_values, L_star_values, time_N2_boundary_values