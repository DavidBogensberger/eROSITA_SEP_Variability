#Code for calculating the bexvar variability from a low count rate light curve.

#Based on bexvar code available at: https://github.com/JohannesBuchner/bexvar

import matplotlib.pyplot as plt
from numpy import sin, log10, log
import numpy as np
import scipy.stats

def lscg_gen(src_counts, bkg_counts, bkg_area, rate_conversion, Ngrid, mincr=0.01):
    """ 
    Generates a log_src_crs_grid applicable to this particular light curve, 
    with appropriately designated limits, for a faster and more accurate 
    run of estimate_source_cr_marginalised and bexvar 
    """
    assert mincr > 0
    logmincr = np.log10(mincr)
    # lowest source count rate possible:
    a = scipy.special.gammaincinv(src_counts + 1, 0.001) / rate_conversion
    # highest count rate possible from background alone:
    b = scipy.special.gammaincinv(bkg_counts + 1, 0.999) / (rate_conversion * bkg_area)
    # put lower limit at the lower end
    m0 = np.nanmax([np.log10(np.min(a - b)), logmincr])
    assert np.isfinite(m0), (m0, np.min(a - b), logmincr, np.nanmax([np.min(a - b), logmincr]))
    # highest source count rate possible:
    m1 = np.log10(np.nanmax(scipy.special.gammaincinv(src_counts + 1, 0.999) / rate_conversion))
    assert np.isfinite(m1), m1
    
    lo = np.nanmax([m0 - 0.05 * (m1 - m0), logmincr])
    hi = m1 + 0.05 * (m1 - m0)
    #print('grid range:', lo, hi, m0, m1, "minsrc:", min(a), "minnetsrc:", min(a-b))
    log_src_crs_grid = np.linspace(lo, hi, Ngrid)

    return log_src_crs_grid

def estimate_source_cr_marginalised(log_src_crs_grid, src_counts, bkg_counts, bkg_area, rate_conversion):
    """ Compute the PDF at positions in log(source count rate)s grid log_src_crs_grid 
    for observing src_counts counts in the source region of size src_area,
    and bkg_counts counts in the background region of size bkg_area.
    
    """
    # background counts give background cr deterministically
    u = np.linspace(0, 1, len(log_src_crs_grid))[1:-1]
    def prob(log_src_cr):
        src_cr = 10**log_src_cr
        bkg_cr = scipy.special.gammaincinv(bkg_counts + 1, u) * bkg_area
        like = scipy.stats.poisson.pmf(src_counts, (src_cr + bkg_cr) * rate_conversion).mean()
        return like
    
    weights = np.array([prob(log_src_cr) for log_src_cr in log_src_crs_grid])
    if weights.sum() == 0:
        print('weights sum = 0,', np.log10(src_counts / rate_conversion))
    weights /= weights.sum()
    
    return weights

def bexvar(log_src_crs_grid, pdfs):
    """ 
    Assumes that the source count rate is log-normal distributed.
    returns posterior samples of the mean and std of that distribution.
    
    pdfs: PDFs for each object 
          defined over the log-source count rate grid log_src_crs_grid.
    
    returns (log_mean, log_std), each an array of posterior samples.
    """
    
    def transform(cube):
        params = cube.copy()
        params[0] = cube[0]*(log_src_crs_grid[-1] - log_src_crs_grid[0]) + log_src_crs_grid[0]
        params[1] = 10**(cube[1]*4 - 2)
        #params[1] = 10**(cube[1]*6 - 4)
        return params

    def loglike(params):
        log_mean  = params[0]
        log_sigma = params[1]
        # compute for each grid log-countrate its probability, according to log_mean, log_sigma
        variance_pdf = scipy.stats.norm.pdf(log_src_crs_grid, log_mean, log_sigma)
	# multiply that probability with the precomputed probabilities (pdfs)
        likes = log((variance_pdf.reshape((1, -1)) * pdfs).mean(axis=1) + 1e-100)
        like = likes.sum()
        if not np.isfinite(like):
            like = -1e300
        return like
    
    from ultranest import ReactiveNestedSampler
    sampler = ReactiveNestedSampler(['logmean', 'logsigma'], loglike, 
        transform=transform, vectorized=False)
    samples = sampler.run(viz_callback=False)['samples']
    log_mean, log_sigma = samples.transpose()
    
    log_mean, log_sigma = samples.transpose()
    
    print(log_mean.mean(), log_mean.std(), log_sigma.mean(), log_sigma.std())
    
    return log_mean, log_sigma

def run_bexvar(src_counts, bkg_counts, bkg_area, fracexp, n_gp):
    log_src_crs_grid = lscg_gen(src_counts, bkg_counts, bkg_area, fracexp, n_gp)

    src_posteriors = []
    for i in range(len(src_counts)):
        pdf = estimate_source_cr_marginalised(log_src_crs_grid, src_counts[i], bkg_counts[i], bkg_area[i], fracexp[i])
        src_posteriors.append(pdf)

    src_posteriors = np.array(src_posteriors)
    logcr_mean, logcr_sigma = bexvar(log_src_crs_grid, src_posteriors)
    return logcr_mean, logcr_sigma
