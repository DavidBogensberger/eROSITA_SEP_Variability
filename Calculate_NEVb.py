#This file contains the functions used to determine the variability of a light curve, using various methods.
#The end of the file shows examples of how to use them, to calculate a NEV, a NEVb, find the corresponding measurement errors, and the intrinsic scatter. 

import numpy as np
import math
from stingray import Lightcurve, Powerspectrum
from bexvar_jbdb import *
from errorbar_calc1 import *

def meanvardet(X):
    #calculates the mean and variance of the list or array "X"
    mX = 0
    for i in range(len(X)):
        mX += X[i] / len(X)
    vX = 0
    for i in range(len(X)):
        vX += (X[i] - mX)**2 / (len(X)-1)
    return mX, vX

def NEVfromPS_PNsub2(T, R, FE, dt, tau):
    #Calculates the NEV by integrating the periodogram
    #It computes the periodgram using Stingray. Make sure they are loaded in. 
    #then subtracts the total Poisson and fractional exposure noise from it.
    #T is an array of times. Make sure these are consistently spaced.
    #R is an array of the count rate.
    #FE is an array of the fractional exposure.
    #dt is a constant denoting the full exposure time of a single bin in a light curve.
    #If the exposure time varies, use FE to denote that variation, and set dt to a useful constant value to compare against. 
    #tau is a constant denoting the separation of bins. This code does not work if the separation of bins varies.
    mRF = np.sum(R*FE) / len(R)
    mf, vf = meanvardet(FE)
    a = 188.48*np.exp(-9.836*mf) + 1.9451
    b = 1159.9*np.exp(-6.174*mf) - 12.74
    fmv = a*vf + b*vf**2
    pnl = (2 * tau / (mRF * dt)) * (1+fmv)
    lc = Lightcurve(T, R, dt=tau, skip_checks=True)
    ps = Powerspectrum(lc, norm='frac')
    df = ps.freq[1] - ps.freq[0]
    NEV = 0
    for i in range(len(ps.power)):
        NEV += df * (ps.power[i] - pnl)
    return NEV

def NEVfromeqweigh(T, R, Re, FE):
    #Finds the NEV and error using a slightly modified version of the standard method
    #It weighs the terms of the NEV equation by the fractional exposure.
    #The error is found from Vaughan+03
    #T is an array of times.
    #R is an array of count rates
    #Re is an array of the errors on the count rate in the format [[r1m, r1p], [r2m, r2p]...], where r1m is the  -1 sigma error for the first data point, and r1p applies to the +1 sigma error. 
    #FE is an array of fractional exposures 
    AR, V, Me2, NEV, FEs, FE2s = 0, 0, 0, 0, 0, 0
    nb0 = len(R)
    for i in range(nb0):
        AR += R[i] * FE[i]
        FEs += FE[i]
        FE2s += FE[i]**2
    if FEs > 0:
        AR /= FEs
    for i in range(nb0):
        V += FE[i]**2 * (R[i]-AR)**2
        if R[i] > AR:
            Me2 += FE[i]**2 * Re[0][i]**2
        else:
            Me2 += FE[i]**2 * Re[1][i]**2
    if FE2s > 0:
        V *= nb0 / ((nb0-1) * FE2s)
        Me2 /= FE2s
    NEV = (V-Me2)/(AR**2)
    NEVe = 0
    if AR > 0:
        neve = ((2*Me2**2)/(nb0*AR**4))+((4*Me2*NEV)/(nb0*AR**2))
        if neve > 0:
            NEVe = math.sqrt(neve)
    return NEV, NEVe

def ARdet(C, BC, BA, FE, dt):
    #Calculates the mean count rate in a light curve.
    #C is an array or list of counts in the source extraction region
    #BC is an array or list of counts in the background extraction region
    #BA is the ratio of the areas of the source to the background extraction region. 
    #FE is an array or list of fractional exposures
    #dt is a constant denoting the full exposure time of a single bin in a light curve.
    #If the exposure time varies, use FE to denote that variation, and set dt to a useful constant value to compare against. 
    AR, fes = 0, 0
    for i in range(len(C)):
        AR += C[i]-BC[i]*BA[i]
        fes += FE[i]*dt
    AR /= fes
    return AR

def NEVfromBV(BV, nb, cr):
    #calculates NEVb from sigma_b, found from bexvar.
    #BV is the sigma_b, the excess variance found by bexvar
    #nb is the number of bins in the light curve
    #cr is the mean count rate, found by using ARdet. 
    a = -0.0296*np.log10(nb) + 0.0363*np.log10(cr) +0.0698
    b = -0.1137*np.log10(nb) + 0.1474*np.log10(cr) +0.796
    c = -0.1036*np.log10(nb) + 0.1580*np.log10(cr) -0.078
    if b**2 - 4 * a * c + 4 * a * np.log10(BV) > 0:
        lnevb = (-b + (b**2 - 4 * a * c + 4 * a * np.log10(BV))**0.5) / (2 * a)
    else:
        lnevb = -1 * b /(2*a)
    NEVb = 10**lnevb
    return NEVb

def NEVefromBVe(BV, BVe, nb, cr):
    #Calculates the measurement error of NEVb, found from NEVfromBV
    #BV is sigma_b, the excess variance found by bexvar
    #BVe is the error in sigma_b, also found from bexvar
    #nb is the number of bins in the light curve
    #cr is the mean count rate, found by using ARdet. 
    y0 =-0.1034*np.log10(nb) - 0.1850*np.log10(cr) -1.410
    a = -0.0296*np.log10(nb) + 0.0363*np.log10(cr) +0.0698
    b = -0.1137*np.log10(nb) + 0.1474*np.log10(cr) +0.796
    c = -0.1036*np.log10(nb) + 0.1580*np.log10(cr) -0.078
    NEVbe = [0, 0]
    if b**2 - 4 * a * c + 4 * a * np.log10(BV) < 0:
        lnev1 = -1 * b / (2*a)
    else:
        lnev1 = (-b + (b**2 - 4 * a * c + 4 * a * np.log10(BV))**0.5) / (2 * a)
    if BVe[0] <= 10**y0:
        NEVbe[0] = -1 * 10**lnev1
    else:
        if b**2 - 4 * a * c + 4 * a * np.log10(BVe[0]) > 0:
            lnev2 = (-b + (b**2 - 4 * a * c + 4 * a * np.log10(BVe[0]))**0.5) / (2 * a)
        else:
            lnev2 = -1 * b / (2*a)
        NEVbe[0] = 10**lnev2 - 10**lnev1
    if b**2 - 4 * a * c + 4 * a * np.log10(BVe[1]) > 0:
        lnev2 = (-b + (b**2 - 4 * a * c + 4 * a * np.log10(BVe[1]))**0.5) / (2 * a)
    else:
        lnev2 = -1 * b / (2*a)
    NEVbe[1] = 10**lnev2 - 10**lnev1
    return NEVbe[0], NEVbe[1]

def IntscattNIV(NEV, Nseg, Nb, adj):
    #The intrinsic scatter in the NIV. It is outputted in the format of an error, and should be treated like a sampling error.
    #NEV is the estimate of the NIV, computed by any of the methods shown above. That can include calculating NEVb, integrating the periodogram, or calculating the difference between variance and the size of the uncertainty.
    #If there is more than one segment, the NEV to be used here is found by taking the geometric mean of the NEV in each segment. 
    #Nseg is the number of segments
    #Nb is the number of bins per segment. Make sure that each segment has an identical number of bins.
    #adj is a parameter to distinguish between instances when the average NEV was computed over adjacent segments (in which case, set adj=1), or over segments that are far apart (in which case, set adj=0)
    #This intrinsic scatter has been calculated under the assumption of a pink noise process. It may not apply to other instances.
    g = 0.648*Nb**(-0.464)+0.2920
    NEVeu = 10**((1-0.1513-((0.1513-1)**2-4*0.0232*(g+np.log10(NEV)))**0.5)/(2*0.0232))
    NEVel = 10**((-1-0.1513+((0.1513+1)**2-4*0.0232*(g-np.log10(NEV)))**0.5)/(2*0.0232))
    NEVe1 = [NEVel - NEV, NEVeu - NEV]
    if adj == 1:
        mf = Nseg**(0.1782*np.log10(NEV)+0.0824)
    else:
        mf = Nseg**(-0.5)
    NEVe = [NEVe1[0]*mf, NEVe1[1]*mf]
    return NEVe
        




#Example of how to use this code, for a single simulated pink noise light curve of 20 bins
Nseg = 1 #Number of segments
Nb = 20 #Number of bins
dt = 40 #duration of each bin
tau = 40 #separation of bins
AR = 3 #Average count rate
ABR = 1 #Average background count rate in background extraction region
BA0 = 0.01 #Size of source to background extraction region.
FEr = [0.1, 0.5] #The range of the fractional exposure. Assume a constant distribution
n_gp = 300
qum = [0.15865, 0.84135]

T = np.arange(0, Nb*tau, tau) #Time of observations
BA = BA0*np.ones(Nb)

#Simulate a pink noise light curve. Use the Timmer and Koenig method:
#Simulate a light curve at least 10 times as long as nb
nf = Nb*5
FS = np.array([(i+1)/(Nb*10) for i in range(nf)]) #Frequencies
PS = FS**(-1) #PSD

def TKlc(PS, nb, minp, maxp):
    #Creates a light curve from the input PSD
    #PS is the power spectrum
    #nb are the number of bins to be simulated.
    #minp and maxp are the minimum and maximum values that the count rate can be.
    C = np.zeros(nb)
    for i in range(len(PS)):
        Ar = math.sqrt(0.5*PS[i]) * np.random.normal() 
        Ai = math.sqrt(0.5*PS[i]) * np.random.normal()
        for j in range(nb):
            C[j] += 2 * Ar * math.cos(2 * math.pi * (i + 1) * j / nb) - 2 * Ai * math.sin(2 * math.pi * (i + 1) * j / nb)
    if max(C) > min(C):
        C = ((maxp - minp) / (max(C) - min(C))) * (C - min(C)) + minp
    return C

Ctlc = TKlc(PS, Nb*10, 0.5, 1.5)
#select a random segment:
sp = np.random.randint(0, Nb*9)
CS = AR * np.array(Ctlc[sp:sp+Nb]) #the true source counts
FE = FEr[0] + (FEr[1] - FEr[0]) * np.random.random(Nb)
C = np.random.poisson((CS + ABR * BA) * FE * dt) #The measured source counts
BC = np.random.poisson(ABR * FE * dt) #The measured background counts

#Compute count rates:
R, Ren, Rep = errbar(C, BC, BA, FE, n_gp, dt) #computes the count rate (R), and the lower (Ren), and upper bound errors (Rep).

#Find the NEV by integrating the light curve:
NEVi = NEVfromPS_PNsub2(T, R, FE, dt, tau)

#Find the FE-weighted NEV by calculating the difference between the variance and the size of the uncertainties:
NEVeqw, NEVeqwme = NEVfromeqweigh(T, R, [Ren, Rep], FE) #calculates the NEV (NEVeqw), and the measurement error (NEVeqwm)

#Find the NEVb, by computing sigma_b from bexvar, then converting it to an estimate of the NIV:
lm, ls = run_bexvar(C, BC, BA, FE, n_gp)
BVm = 10**(np.array([math.log10(ls[p]) for p in range(len(ls))]).mean()) #sigma_b
bs = sorted(ls)
bsl, bsu = bs[math.floor(qum[0]*(len(bs)-1))], bs[math.floor(qum[1]*(len(bs)-1))] #lower and upper bounds. 
#Determine average count rate:
AR = ARdet(C, BC, BA, FE, dt)
NEVb = NEVfromBV(BVm, Nb, AR)      #the conversion function from sigma_b to NIV
NEVbmem, NEVbmep = NEVefromBVe(BVm, [bsl, bsu], Nb, AR)      #Calculates the measurement error (NEVbmem in the negative direction, NEVbmep in the positive direction)
NEVbsem, NEVbsep = IntscattNIV(NEVb, Nseg, Nb, 1)      #Calculates the intrinsic scatter in the estimate of the NIV. Again, NEVbsem is the error in the negative direction, NEVbsep is in the positive direction.
NEVbtem, NEVbtep = -1*np.sqrt(NEVbmem**2+NEVbsem**2), np.sqrt(NEVbmep**2+NEVbsep**2)      #the total error. To be used when comparing NEVb measurements for different sources, or for the same source at different times. Includes the measurement error and the intrinsic scatter, which acts like a sampling error.

print(AR, NEVi, NEVeqw, NEVeqwme, BVm, NEVb, [NEVbmem, NEVbmep], [NEVbsem, NEVbsep], [NEVbtem, NEVbtep])

input('Press enter to continue')




#Second example of how to use this code, for a simulated pink noise light curve of 5 adjacent segments of 20 bins each. 
Nseg = 5 #Number of segments
Nb = 20 #Number of bins
dt = 40 #duration of each bin
tau = 40 #separation of bins
AR = 3 #Average count rate
ABR = 1 #Average background count rate in background extraction region
BA0 = 0.01 #Size of source to background extraction region.
FEr = [0.1, 0.5] #The range of the fractional exposure. Assume a constant distribution
n_gp = 300

T = [np.arange(0, Nb, 1) for i in range(Nseg)] #Time of observations
BA = [BA0*np.ones(Nb) for i in range(Nseg)]

#Simulate a pink noise light curve. Use the Timmer and Koenig method:
#Simulate a light curve at least 10 times as long as nb
nf = Nb * Nseg * 5
FS = np.array([(i + 1)/(nf * 2) for i in range(nf)]) #Frequencies
PS = FS ** (-1) #PSD

Ctlc = TKlc(PS, Nb * Nseg * 10, 0.5, 1.5)
#select a random segment:
sp = np.random.randint(0, Nb * Nseg * 9)
CS = [AR * np.array(Ctlc[sp + i * Nb : sp + (i+1) * Nb]) for i in range(Nseg)] #the true source counts
FE = [FEr[0] + (FEr[1] - FEr[0]) * np.random.random(Nb) for i in range(Nseg)]
C = [np.random.poisson((CS[i] + ABR * BA[i]) * FE[i] * dt) for i in range(Nseg)] #The measured background counts
BC = [np.random.poisson(ABR * FE[i] * dt) for i in range(Nseg)] #The measured background counts

#This time only compute the bexvar NEVb:
NEVb, NEVbme = [0]*Nseg, [[0,0] for i in range(Nseg)]
for i in range(Nseg):
    #Compute count rates:
    R, Ren, Rep = errbar(C[i], BC[i], BA[i], FE[i], n_gp, dt) #computes the count rate (R), and the lower (Ren), and upper bound errors (Rep).
    #Find the NEVb, by computing sigma_b from bexvar, then converting it to an estimate of the NIV:
    lm, ls = run_bexvar(C[i], BC[i], BA[i], FE[i], n_gp)
    BVm = 10**(np.array([math.log10(ls[p]) for p in range(len(ls))]).mean()) #sigma_b
    bs = sorted(ls)
    bsl, bsu = bs[math.floor(qum[0]*(len(bs)-1))], bs[math.floor(qum[1]*(len(bs)-1))] #lower and upper bounds. 
    #Determine average count rate:
    AR = ARdet(C[i], BC[i], BA[i], FE[i], dt)
    NEVb[i] = NEVfromBV(BVm, Nb, AR)      #the conversion function from sigma_b to NIV
    NEVbme[i][0], NEVbme[i][1] = NEVefromBVe(BVm, [bsl, bsu], Nb, AR)      #Calculates the measurement error (NEVbmem in the negative direction, NEVbmep in the positive direction)
    NEVbme[i][0] *= -1 #to ensure the sign is positive. 

#Calculate the geometric mean NEVb:
gmNEVb = 10**np.sum([np.log10(NEVb[i])/Nseg for i in range(Nseg)])
gmNEVbme = [10**np.sum([np.log10(NEVbme[i][0])/Nseg for i in range(Nseg)]), 10**np.sum([np.log10(NEVbme[i][1])/Nseg for i in range(Nseg)])]
#Now calculate the intrinsic scatter for it:
gmNEVbse = IntscattNIV(gmNEVb, Nseg, Nb, 1)
#And the total error, for comparing different NIV estimates:
gmNEVbte = [-1*np.sqrt(gmNEVbme[0]**2+gmNEVbse[0]**2), np.sqrt(gmNEVbme[1]**2+gmNEVbse[1]**2)]

print(NEVb, NEVbme, gmNEVb, gmNEVbme, gmNEVbse, gmNEVbte)





