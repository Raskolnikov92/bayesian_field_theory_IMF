#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 18:59:12 2021

@author: atsouros
"""

import sys
import numpy as np
import nifty7 as ift
import random as rn
import itertools
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.axes_grid1 import make_axes_locatable


def random_los(n_los): 
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends

def radial_los(n_los):
    starts = list(ift.random.current_rng().random((n_los, 2)).T)
    ends = list(0.5 + 0*ift.random.current_rng().random((n_los, 2)).T)
    return starts, ends

def make_random_mask(domain):
    mask = ift.from_random(domain, 'pm1') #This creates a field whose values are either -1 or 1 on each pixel of the input domain
    #'pm1' stands for plus-minus 1
    mask = (mask + 1)/2 #this maps -1->0 and 1->1
    return mask.val

def make_perc_mask(N):
    #this function does the same as make_random_mask, but you get to chose the percentage of pixels you want to keep.
    frac = 0.01 #percentage of pixels kept.
    arr = np.asarray([0,1])
    mat = np.zeros(shape = (N,N))
    for i,j in itertools.product(range(N),range(N)):
        mat[i][j] = int(np.asarray(rn.choices(arr, weights=[frac, 1 - frac]))) 
    return mat

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def rms(arr):
    """
    Returns RMS of array elements.
    """
    return np.sqrt(np.mean(np.square(arr)))

def divisors(n):
    """
    Returns an array of all the divisors of the integer n.
    """
    if not(isinstance(n,int)):
        raise ValueError('The input of the divisors function must be an integer.')
        
    ret = []
    for i in range(1,n-1):
        if (n % (i+1)) == 0:
            ret.append(i+1)
    if not ret:
        raise ValueError('Number of pixels per axis is prime. Choose a more composite number.')
    else:
        return np.array(ret)
    
def plots(mat1, mat2, mat3):
    cmap = 'plasma'
    fig, axes = plt.subplots(1, 3, figsize=(15,15))
    pl1 = axes[0].imshow(mat1, cmap=cmap)
    
    
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(pl1, cax=cax, orientation='vertical')
    
    pl2 = axes[1].imshow(mat2, cmap=cmap)
    
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(pl2, cax=cax, orientation='vertical')
    
    
    pl3 = axes[2].imshow(mat3, cmap=cmap)
    
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(pl3, cax=cax, orientation='vertical')
    
    plt.show()
    return 

#In order to input point data, perhaps check from_raw instead of from_random    

def main():    
    #mode = 0,1,2 for random LoS, radial LoS, and masked pixels respectively. Default is radial LoS
    if len(sys.argv) == 2:
        mode = int(sys.argv[1])
    else:
        mode = 2
    #Here I define the signal space ()
    N_pixels = 360 #number of pixels in N_pixelsxN_pixels grid    
    if not(mode in [0,1,2]):
        raise ValueError('Mode must be either 0, 1, or 2.')
        
    position_space = ift.RGSpace([N_pixels, N_pixels])
        
    
    # Specify harmonic space corresponding to signal space
    harmonic_space = position_space.get_default_codomain()
    
    # Harmonic transform from harmonic space to position space
    HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)
    
    # Set prior correlation covariance with a power spectrum leading to
    # homogeneous and isotropic statistics
    def power_spectrum(k):
        s = 3
        output = []
        kc = 1
        c = kc**(-s)
        if isinstance(k,np.ndarray):
            for ki in k:
                if ki<= kc:
                    output.append(c)
                else:
                    output.append(ki**(-s))
            output = np.asarray(output)
        else:
            if k<=kc:
                return c
            else:
                return k**(-s)       
        return output
    
    plot_power_spectrum = False
    if plot_power_spectrum:
        x = np.linspace(0,100,100)
        y = [power_spectrum(i) for i in x]
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(x,y)    
        plt.show()
    # 1D spectral space on which the power spectrum is defined
    power_space = ift.PowerSpace(harmonic_space)
    
    # Mapping to (higher dimensional) harmonic space
    PD = ift.PowerDistributor(harmonic_space, power_space)
    
    # Apply the mapping
    prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))
    
    # Insert the result into the diagonal of an harmonic space operator
    S = ift.DiagonalOperator(prior_correlation_structure)
    #S is the prior field covariance. It is diagonal (assuming homogeneity and isotropy).
    
    #Now we build the response operator
    
    # Masking operator to model that parts of the field have not been observed
    #mask = make_random_mask(position_space)
    
    noise = 10. #noise covariance
    
    if mode == 0:
        N_LoS = 5000
        LOS_starts, LOS_ends = random_los(N_LoS)
        MOCK_SIGNAL = S.draw_sample_with_dtype(dtype=np.float64)
        r = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
        R = r(HT)
        data = R(MOCK_SIGNAL)
        
       
        # Specify noise
        data_space = R.target
        N = ift.ScalingOperator(data_space, noise)
        
        # Create mock data
        MOCK_NOISE = N.draw_sample_with_dtype(dtype=np.float64)
        
        # Generate mock signal and data
        data = R(MOCK_SIGNAL) + MOCK_NOISE
    elif mode == 1:
        N_LoS = 5000
        LOS_starts, LOS_ends = radial_los(N_LoS)
        MOCK_SIGNAL = S.draw_sample_with_dtype(dtype=np.float64)
        r = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
        R = r(HT)
        data = R(MOCK_SIGNAL)
        
       
        # Specify noise
        data_space = R.target
        N = ift.ScalingOperator(data_space, noise)
        
        # Create mock data
        MOCK_NOISE = N.draw_sample_with_dtype(dtype=np.float64)
        
        # Generate mock signal and data
        data = R(MOCK_SIGNAL) + MOCK_NOISE
    elif mode == 2:
        mask = make_perc_mask(N_pixels)
        mask = ift.Field.from_raw(position_space, mask)
        Mask = ift.MaskOperator(mask)
        
        R = Mask(HT)
        
        data_space = R.target
        
        N = ift.ScalingOperator(data_space, noise)
        
        # Create mock data
        MOCK_SIGNAL = S.draw_sample_with_dtype(dtype=np.float64)
        MOCK_NOISE = N.draw_sample_with_dtype(dtype=np.float64)
        data = R(MOCK_SIGNAL) + MOCK_NOISE
        
    # Build inverse propagator D and information source j
    D_inv = R.adjoint @ N.inverse @ R + S.inverse
    j = R.adjoint_times(N.inverse_times(data))
    # Make D_inv invertible (via Conjugate Gradient)
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
    
    # Calculate Wiener filter solution
    m = D(j)

    # Plotting
    rg = isinstance(position_space, ift.RGSpace)
    plot = ift.Plot()
    
    cmap = None
    
    plot_histogram = True #True draws histogram of residuals and data
    
    if plot_histogram:  
        x = HT(MOCK_SIGNAL).val - HT(m).val
        x = x.flatten(order='C')
        x_signal = - HT(MOCK_SIGNAL).val
        x_signal = x_signal.flatten(order='C')
        plt.hist(x, bins = 'auto', density = True, histtype='step', color = 'b')
        plt.hist(x, bins = 'auto', density = True, histtype='stepfilled', alpha = 0.3, color = 'b', label = 'Residuals')
        plt.hist(x_signal, bins = 'auto', density = True, histtype='step', color = 'r')
        plt.hist(x_signal, bins = 'auto', density = True, histtype='stepfilled', alpha = 0.3, color = 'r', label = 'Data')
        plt.legend()
        plt.show()
    
    average_pixels = True #True if you want to plot the result of sequentially averaging pixels and comparing results. Also shows change of RMS(Residual)/RMS(signal) wrt to length scale
    
    if average_pixels:
        mat1 = HT(MOCK_SIGNAL).val
        mat2 = HT(m).val
        mat3 = HT(MOCK_SIGNAL).val - HT(m).val
    
        N = N_pixels
        x = [1/N]
        y = [rms(mat1.flatten(order='C'))/rms(mat2.flatten(order='C'))]  
        plots(mat1, mat2, mat3)
        for i in divisors(N):
            x.append(i/N)
            reduced_mat1 = np.zeros((N//i,N//i))
            reduced_mat2 = np.zeros((N//i,N//i))
            reduced_mat3 = np.zeros((N//i,N//i))
            s = 0
            for j, k in itertools.product(range(N//i), range(N//i)):
                reduced_mat1[j][k] = np.mean(blockshaped(mat1, i, i)[s].flatten(order='C'))
                reduced_mat2[j][k] = np.mean(blockshaped(mat2, i, i)[s].flatten(order='C'))
                reduced_mat3[j][k] = np.mean(blockshaped(mat3, i, i)[s].flatten(order='C'))
                s += 1
            y.append(rms(reduced_mat1.flatten(order='C'))/rms(reduced_mat2.flatten(order='C')))
            plots(reduced_mat1, reduced_mat2, reduced_mat3)
            
        plt.plot(x,y,'ko')    
    
    filename = "nifty_test_config_space.png"
   
    plot.add(HT(MOCK_SIGNAL), title='Mock Signal', cmap = cmap)
    if mode == 0 or mode == 1:
        plot.add(r.adjoint(data), title='Data', cmap = cmap)
    elif mode == 2 :
        plot.add(Mask.adjoint(data), title='Data', cmap = cmap)
    plot.add(HT(m), title='Reconstruction', cmap = cmap)
    plot.add(HT(MOCK_SIGNAL) - HT(m), title='Residuals', cmap = cmap) #x = RMS of residuals/RMS of map (single number) -> smoothing  at different length scales, and   
    plot.output(nx=2, ny=2, xsize=10, ysize=10, name=filename)
    print("Saved results as '{}'.".format(filename))
    
    

if __name__ == '__main__':
    main()