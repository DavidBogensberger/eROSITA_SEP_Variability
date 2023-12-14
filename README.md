# eROSITA_SEP_Variability
A selection of codes for identifying, quantifying, and analysing variability for X-ray detected sources in the South Ecliptic Pole with eROSITA.

bexvar_jbdb.py, and errorbar_calc1.py create functions that are imported by other codes. bexvar_jbdb.py is used for estimating the bexvar log variance. errorbar_calc1.py is used to calculate the error bars for eROSITA light curves. 

Powerspectrum_eR123_SEP.py computes the power spectrum for a set of input light curves. These are combined, and then split into segments within which the obsevations are almost continuous (with one bin every 4 hours). The power spectrum is computed for each segment individually. They are then averaged into one powerspectrum. We subtract the Poisson Noise level, which is different to its standard form due to the variation in the fractional exposure. We fit the resulting averaged powerspectrum with an aliased broken powerlaw. The aliasing component is necessary due to the long, but consistant gaps between short, regular observations. 

Identify_variable_sources.py determines whether a set of input light curves are variable, or not. Each light curve is evaluated on the grid of 1, 2, 3 sigma variability boundaries on the two parameters SCATT_LO, and AMPL_SIG. 1, 2, 3 sigma refers here to what fraction of intrinsically non-variable sources are excluded (84.1%, 97.7%, and 99.87%, respectively). We chose a 3 sigma boundary as the discriminating condition between variable and non-variable sources, as that reduced the false positive rate to 0.13%, while still retaining a lot of variable sources. These boundaries depend on both the count rate and the number of bins. The script outputs a fits file that contains information about the SCATT_LO and AMPL_SIG variability classification. If either of those are found to be above 3 sigma, then the variability classification for that source is set to 1. Otherwise it is 0. This file also requires the data contained in the text files bv_varnonvar2boundaries_nrep2000_all.txt, and mad_varnonvar2boundaries_nrep2000_all.txt.

Calculate_NEVb calculates NEVb, NEVi, and NEVeq, and provides the measurement errors, as well as the intrinsic scatter in the NIV. It shows how to estimate the NIV_infty, for comparing the variability strength of different sources, or the same source at different times, by combining the measurement errors and the intrinsic scatter in the NIV. It can be used to investigate light curves with fixed, or varying exposure times, fixed, or varying background areas and count rates, and any number of source counts per bin, whether in the Poisson regime, or not. 

Lightcurvegeneration produces light curve graphs from eROSITA lightcurve files
