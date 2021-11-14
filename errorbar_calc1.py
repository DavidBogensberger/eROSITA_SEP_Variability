"""
This code can be used to determine Bayesian errorbars, that are not of equal lenght in the + and - directions, do not extend below 0, and are an accurate description of the interval that has a ~68% likelihood of containing the true input count rate, for given measured number of source counts, background counts, background ratio, time bin size, and fractional exposure.

These errorbars can be used to obtain reasonable estimates of the variability of sources through measures like the normalized excess variance. However, they can give wrong results for low count rates, for which other methods like bexvar should be used. The errorbars calculated in this way agree with the eSASS errorbars in the high count rate regime.

When rebinning data, do not propagate errors found from this method to determine the error bars for the rebinned light curve. Instead, run this code again for the new bins. 
"""

import numpy as np
import scipy.stats

def lscg_gen(src_counts, bkg_counts, bkg_area, fracexp, n_gp):
    """ 
    Generates a log_src_crs_grid applicable to this particular light curve, 
    with appropriately designated limits, for a faster and more accurate 
    run of estimate_source_cr_marginalised and bexvar 
    """
    a, b = scipy.special.gammaincinv(src_counts/fracexp +1, 0.001), scipy.special.gammaincinv((bkg_counts * bkg_area)/ fracexp  + 1, 0.999)
    if a - b < 0:
        m0 = -1
    else:
        m0 = np.log10(a - b)
    m1 = np.log10(scipy.special.gammaincinv(src_counts/fracexp + 1, 0.999))
    
    if m0 - 0.05 * (m1-m0) < -1:
        log_src_crs_grid = np.linspace(-1.0, m1 + 0.05*(m1 - m0), n_gp)
    else:
        log_src_crs_grid = np.linspace(m0 - 0.05 * (m1 - m0), m1 + 0.05*(m1 - m0), n_gp)
        
    return log_src_crs_grid

def estimate_source_cr_marginalised(log_src_crs_grid, src_counts, bkg_counts, bkg_area, fracexp):
    """ Compute the PDF at positions in log(source count rate)s grid log_src_crs_grid 
    for observing src_counts counts in the source region,
    and bkg_counts counts in the background region of size 1/bkg_area relative to
    the source region, with a fractional exposure of fracexp.
    
    """
    # background counts give background cr deterministically
    u = np.linspace(0, 1, len(log_src_crs_grid))[1:-1]
    def prob(log_src_cr):
        src_cr = 10**log_src_cr
        bkg_cr = scipy.special.gammaincinv(bkg_counts + 1, u) * bkg_area
        like = scipy.stats.poisson.pmf(src_counts, (src_cr + bkg_cr) * fracexp).mean()
        return like
    
    weights = np.array([prob(log_src_cr) for log_src_cr in log_src_crs_grid])
    if weights.sum() == 0:
        print(np.log10(src_counts / fracexp))
    weights /= weights.sum()
    
    return weights

def errbar(src_counts, bkg_counts, bkg_area, fracexp, n_gp, dt):
    N = len(src_counts)
    R, Ren, Rep = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        log_src_crs_grid = lscg_gen(src_counts[i], bkg_counts[i], bkg_area[i], fracexp[i], n_gp)
        pdf = estimate_source_cr_marginalised(log_src_crs_grid, src_counts[i], bkg_counts[i], bkg_area[i], fracexp[i])
        
        mid = np.argmax(pdf) 
        upper = mid + np.where(pdf[mid:] < pdf[mid] / np.exp(0.5))[0] 
        lower = mid - np.where(pdf[:mid][::-1] < pdf[mid] / np.exp(0.5))[0]
        if len(lower) > 0: 
            lower_value, mid_value, upper_value = 10**log_src_crs_grid[[lower[0], mid, upper[0]]] 
        else: 
            lower_value = 0
            mid_value, upper_value = 10**log_src_crs_grid[[mid, upper[0]]] 

        #Convert to a count rate
        R[i] = mid_value / dt
        Rep[i] = upper_value / dt - R[i]
        Ren[i] = R[i] - lower_value / dt

    return R, Ren, Rep


#Here is an example run of the errorbar determination:

#nb = 30           #Number of data points
#dt = 40           #Time bin size (s)
#Dt = 4 * 60 * 60  #Separation between time bins
#T0 = 0            #Time of the first observation
#minFE = 0.1       #The minimum fractional exposure accepted for each bin
#n_gp = 100        #Number of grid points to be used 

##Simulate a light curve:

#T = np.arange(T0, T0 + nb * Dt, Dt)              #Time of bin
#ci = 10**(np.random.normal(np.log10(5), 0.3, nb))#True count rate * dt
#bci = 20 * np.ones(nb)                           #True back count rate * dt
#bkg_area = 0.01 * np.ones(nb)                    #Ratio of src to bkg area
#fracexp = 0.667 * np.random.random(nb)           #Fractional exposure
#src_counts = np.random.poisson(fracexp * (ci + bkg_area * bci), nb)
                                                 #Measured source counts
#bkg_counts = np.random.poisson(fracexp * bci, nb)
                                                 #Measured background counts

##Remove bins with fractional exposure < minFE:

#k = 0
#while k < len(T):
#    if fracexp[k] < minFE:
#        ci, bci, bkg_area, fracexp, src_counts, bkg_counts, T = np.delete(ci, k), np.delete(bci, k), np.delete(bkg_area, k), np.delete(fracexp, k), np.delete(src_counts, k), np.delete(bkg_counts, k), np.delete(T, k)
#    else:
#        k += 1

#Determine error bars for this light curve:

#R, Ren, Rep = errbar(src_counts, bkg_counts, bkg_area, fracexp, n_gp, dt)

#Plot the light curve with the new errorbars:

#import matplotlib
#import matplotlib.pyplot as plt

#matplotlib.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'
#matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

#fig, ax = plt.subplots(1,1, figsize=(10,6))
#ax.errorbar(T, R, yerr=[Ren, Rep], fmt='o', color='green', markersize=4, elinewidth=1)
#ax.set_xlabel('Time (s)')
#ax.set_ylabel('Count rate (cts/s)')
#ax.set_title('Light Curve')
#name = 'errorbar_calc_lightcurve_with_bayesian_errorbars.png'
#plt.savefig(name, format='png')
#plt.close()
