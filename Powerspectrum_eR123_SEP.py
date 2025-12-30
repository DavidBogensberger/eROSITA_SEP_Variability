#This code computes the powerspectrum of a selected eROSITA X-ray source close to the SEP.

#Input the eROSITA light curves across multiple eRASS.
#The code selects segments within those eRASS Light curves, for which there are (almost) continuous observations (one eroday every 4 hours).
#It computes the powerspectrum in each of the segments, and then averages them.

#It deals with an enhanced Poisson Noise level due to a varying fractional exposure, and deals with the aliasing effects of having long gaps between observations.

#It tries to fit a broken powerlaw to the averaged powerspectru, which is smoothed with a Gaussian smoothing algorithm. 

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib
import os

from errorbar_calc1 import *
from stingray import Lightcurve, Powerspectrum, AveragedPowerspectrum

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
plt.rcParams.update({'font.size': 13})

def meanvardet(X):
    mX = 0
    for i in range(len(X)):
        mX += X[i] / len(X)
    vX = 0
    for i in range(len(X)):
        vX += (X[i] - mX)**2 / (len(X)-1)
    return mX, vX

def PoissonLevel(C, m, v):
    AR = 0
    for i in range(len(C)):
        AR += C[i] / (len(C)*dt*rbf)
    p1 = 1.78 + 166*np.exp(-9.59*m)
    p2 = -9.9 + 2495*np.exp(-6.88*m)
    PL = (p2*v**2 + p1*v + 1)* 2 /(AR * m)
    return PL

def alipowl(fx, nc, ps1, ps2, bf):
    #fits the powerlaw with an aliasing function.
    #Need to specify ef and sp before running this function
    #sf = sampling frequency
    #ef = end frequency. Above this frequency there is no more power
    P = np.zeros(len(fx))
    for i in range(len(fx)):
        #first cover the true power, and the aliased power with f +k sf
        f = fx[i]
        while f < ef:
            if f < bf:
                P[i] += nc * f**ps1
            else:
                P[i] += nc * bf**(ps1-ps2) * f**ps2
            f += sf
        #And now for the k sf - f part:
        f = sf - fx[i]
        while f < ef:
            if f < bf:
                P[i] += nc * f**ps1
            else:
                P[i] += nc * bf**(ps1-ps2) * f**ps2
            f += sf
    lP = np.log10(P)
    return lP

def Gaussmooth(x, y, sigma):
    #Smoothes y, based on the values x.  
    lx = np.log10(x)
    sy = np.zeros(len(y))
    for i in range(len(x)):
        gs = 0 # the sum of the gaussian elements
        sy1 = 0 # the sum of all terms. 
        #find the instances that are within 3 sigma in lx from a particular x. 
        for j in range(len(x)):
            if lx[j] - lx[i] > 3 * sigma:
                break
            if abs(lx[i] - lx[j]) < 3 * sigma:
                gs += np.exp(-((lx[i] - lx[j])**2/(2*sigma**2)))
                sy1 += np.exp(-((lx[i] - lx[j])**2/(2*sigma**2))) * y[j]
        if gs > 0:
            sy[i] = sy1 / gs
        else:
            print('Problem in Gaussmooth, ', i, lx[i], gs, sy1)
    return sy

#Input parameters:
m = 3 #This corresponds to the 0.2-5.0 keV energy band. 
dt = 4*60*60
sf = 1/dt
ef = 0.01
minFE = 0.1
n_gp = 300
rbf = 1 # the number of erodays that should be binned together.

LCfiles = [] #write the list of fits files to be used here. e.g.
#LCfiles = ['em01_093156_020_020_LightCurve_00004_c946.fits', 'em02_093156_020_020_LightCurve_00007_c946.fits', 'em03_093156_020_020_LightCurve_00005_c946.fits']

T, C, BC, BA, FE = [], [], [], [], []

for r in range(len(LCfiles)):
    print('Working on light curve: ', LCfiles[r])
    if os.path.isfile(LCfiles[r]) == 0:
        print('\n', LCfiles[r], 'DOES NOT EXIST!\n')
    F = fits.open(LCfiles[r])
    Tmjdref = F[1].header["MJDREF"]
    T0 = F[1].data["Time"]
    C0 = F[1].data["Counts"]
    BC0 = F[1].data["Back_counts"]
    FE0 = F[1].data["Fracexp"]
    BA0 = F[1].data["backratio"]
    F.close()
    n = len(T0)

    Tmjd = [Tmjdref+(T0[j]/(24*60*60)) for j in range(n)]
    ise_erd = [[0]]
    j = 0
    for k in range(n):
        if T0[k]-T0[ise_erd[j][0]] > (4*rbf - 2)*60*60:
            ise_erd[j].append(k)
            ise_erd.append([k])
            j+=1
    Ned = len(ise_erd)
    ise_erd[Ned-1].append(n)
    
    t = [0]*Ned
    c = [0]*Ned
    bc = [0]*Ned
    fe = [0]*Ned
    ba = [0]*Ned

    for l in range(Ned):
        nbped = 0
        for k in range(ise_erd[l][0], ise_erd[l][1]):
            if FE0[k][m] > 0:
                if np.isinf(C0[k][m]) + np.isnan(C0[k][m]) + np.isinf(BC0[k][m]) + np.isnan(BC0[k][m]) + np.isinf(FE0[k][m]) + np.isnan(FE0[k][m]) + np.isinf(BA0[k]) + np.isnan(BA0[k])  == 0:
                    c[l] += C0[k][m]
                    bc[l] += BC0[k][m]
                    fe[l] += FE0[k][m]
                    ba[l] += BA0[k]
                    t[l] += T0[k] * FE0[k][m]
                    nbped += 1
        if nbped > 0:
            ba[l] /= nbped
        if fe[l] > 0:
            t[l] /= fe[l]

    #Now delete the bins that have a too small FE.
    u = 0
    while u < len(fe):
        if fe[u] < minFE:
            del c[u], bc[u], fe[u], ba[u], t[u]
        else:
            u += 1

    #Now append all of these to T, C, ...
    for j in range(len(t)):
        T.append(t[j])
        C.append(c[j])
        BC.append(bc[j])
        BA.append(ba[j])
        FE.append(fe[j])

#Now split into different segments.
ns = 1
js = [[0]]
for j in range(1, len(T)):
    if T[j]-T[j-1] > 5*24*60*60:
        ns += 1
        js[-1].append(j)
        js.append([j])
js[-1].append(len(T)-1)

#Only take segments that are at least 20 bins long.
k = 0
while k < len(js):
    if js[k][1] - js[k][0] < 20:
        del js[k]
        ns -= 1
    else:
        k += 1
print('Number of segments:', ns)

#Split the data into these segments:
Ts, Cs, BCs, BAs, FEs = [[] for i in range(ns)], [[] for i in range(ns)], [[] for i in range(ns)], [[] for i in range(ns)], [[] for i in range(ns)]
for i in range(ns):
    for j in range(js[i][0], js[i][1]):
        Ts[i].append(T[j])
        Cs[i].append(C[j])
        BCs[i].append(BC[j])
        BAs[i].append(BA[j])
        FEs[i].append(FE[j])

#Now generate the power spectra for these.
FS, PS = [0]*ns, [0]*ns

for r in range(ns):
    Rs1, Rems, Reps = errbar(np.array(Cs[r]), np.array(BCs[r]), np.array(BAs[r]), np.array(FEs[r]), n_gp, dt*rbf)
    Rs = Rs1.copy()*dt*rbf
    lc = Lightcurve(Ts[r], Rs, dt=dt)
    ps = Powerspectrum(lc, norm='frac')
    FS[r], PS[r] = ps.freq, [0]*len(ps.power)
    mF, vF = meanvardet(FEs[r])
    pl = PoissonLevel(Rs, mF, vF)
    for i in range(len(ps.power)): 
        PS[r][i] = ps.power[i] - pl

#Average the power spectra:
FA, PA = [], []

minf = min([min(FS[r]) for r in range(ns)])
ned = [len(Ts[r]) for r in range(ns)]

for r in range(ns):
    for i in range(len(FS[r])):
        #determine if frequency is already contained in FA: 
        k = 0
        for j in range(len(FA)):
            if (FS[r][i] - FA[j])**2 < (0.0001*minf)**2:
                k = 1
        if k == 0:
            FA.append(FS[r][i])
            sw = ned[r] # weigh the average by the number of bins. 
            sp = PS[r][i] * ned[r] # summed weighted power
            #Now for the other terms from the other segments:
            for u in range(ns):
                if u == r:
                    continue   
                if FA[-1] < min(FS[u]):
                    sw += 0
                else:
                    if FA[-1] > max(FS[u]):
                        sw += 0
                    else:
                        #Determine the two datapoints on either side. 
                        k = 0
                        while FS[u][k] < FA[-1]:
                            k += 1
                        #determine the value at that point:
                        #assume a linear relation
                        pta = PS[u][k-1] + ((FA[-1] - FS[u][k-1]) * (PS[u][k] - PS[u][k-1]) / (FS[u][k]-FS[u][k-1]))
                        sw += ned[u]
                        sp += pta * ned[u]
            PA.append(sp / sw)

PA = [x for _,x in sorted(zip(FA, PA))]
FA = sorted(FA)

#Smooth the power spectrum with a Gaussian smoothing algorithm:
PAgs = Gaussmooth(FA, PA, 0.05)

#Remove the bins that contain negative power
FA, PA, PAgs = np.array(FA), np.array(PA), np.array(PAgs)
FA, PA, PAgs = FA.tolist(), PA.tolist(), PAgs.tolist()
FAgs = FA.copy()

k = 0
while k < len(PAgs):
    if PAgs[k] < 0:
        del PAgs[k], FAgs[k]
    else:
        k += 1

#Now fit the data with the aliased broken powerlaw:
inp = [1, -1, -2, 2e-4]
pmin = [0.00001, -2, -3, 1e-7]
pmax = [100000, 0, 0, 1e-2]

f, fc = scipy.optimize.curve_fit(alipowl, FAgs, np.log10(PAgs), p0=inp, bounds=(pmin, pmax), maxfev=10000)
Perr = np.sqrt(np.diag(fc))
print('\nBest fit parameters, for fitting the power spectrum with an aliased broken powerlaw:', f)
print('Errors in fit parameters:', Perr)

yp = 10**alipowl(FAgs, f[0], f[1], f[2], f[3])

fig, ax = plt.subplots(1,1, figsize=(10,6))
ax.plot(FAgs, yp, color='blue', linewidth=1, linestyle='--', label='Best fit')
ax.plot(FAgs, PAgs, color='red', linewidth=3, label='Gaussian smoothed average power spectrum')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power')
ax.set_title('Fitted Gaussian smoothed Averaged Power Spectrum')
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(1.5)
ax.tick_params(length=7, width=1, which='major', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
ax.tick_params(length=3.5, width=1, which='minor', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
plt.legend(prop={'size':12})
plt.savefig('APSfit_aliasbrokpl.png', format='png')

plt.show()

