# eROSITA_SEP_Variability
A selection of codes for identifying, quantifying, and analysing variability for X-ray detected sources in the South Ecliptic Pole with eROSITA.

bexvar_jbdb.py, and errorbar_calc1.py create functions that are imported by other codes. bexvar_jbdb.py is used for estimating the bexvar log variance. errorbar_calc1.py is used to calculate the error bars for eROSITA light curves. 

Powerspectrum_eR123_SEP.py computes the power spectrum for a set of input light curves. These are combined, and then split into segments within which the obsevations are almost continuous (with one bin every 4 hours). The power spectrum is computed for each segment individually. They are then averaged into one powerspectrum. We subtract the Poisson Noise level, which is different to its standard form due to the variation in the fractional exposure. We fit the resulting averaged powerspectrum with an aliased broken powerlaw. The aliasing component is necessary due to the long, but consistant gaps between short, regular observations. 

