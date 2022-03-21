#This code identifies variable sources from a set of input light curves, using the two methods SCATT_LO (the 10% quantile of the bexvar sigma posterior), and AMPL_SIG (the significance of the maximum amplitude deviation).

#These two methods have been found to be best suited for distinguishing between variable and non-variable sources.

#Neither of these methods cares about the time ordering of the light curve. It makes no difference if all the observations occur one eroday after another, or if there are half year gaps in the dataset. The more datapoints, the more accurate the distinction is. 

#Results of the variability analysis will be saved to a fits file, or which the columns 'Variability_Category_SCATT_LO', and 'Variability_AMPL_SIG' can each individually have values of integers between 0 and 3. A value of 0 means that the source's light curve lies below the 1 sigma boundary. A value of 1 means that it lies between 1 and 2 sigma. A value of 2 means that it lies between 2 and 3 sigma. And a value of 3 means that it is in excess of 3 sigma.

#Finally, the column 'Variability_Category' can either have a value of 0, or 1. 1 corresponds to the particular source being variable in excess of 3 sigma in at least one of the two criteria SCATT_LO or AMPL_SIG

#For this code to work, the directory of this file needs to also contain the two txt files: 'bv_varnonvar2boundaries_nrep2000_all.txt', and 'mad_varnonvar2boundaries_nrep2000_all.txt'

import numpy as np
from astropy.io import fits
import math
from astropy.table import Table
import os

from bexvar_jbdb import *
from errorbar_calc1 import *

#to be inputted: 
eb = 3 #the energy band for which the variability analysis is performed.

#Specify input light curves here:
#e.g.:
#lcf = ['em01_093156_020_020_LightCurve_00004_c946.fits', 'em02_093156_020_020_LightCurve_00007_c946.fits', 'em03_093156_020_020_LightCurve_00005_c946.fits']
lcf = []
Nlc = len(lcf)


minFE = 0.1 # the minimum fractional exposure. Bins with smaller fractional exposure are removed. 

n_gp = 300
dted = 40 #The length of an eroday observation


def findi(c, n, Cl, Nl): # find i. the Cl, Nl are the lists that contain all the permutations of C and N
    ln = len(Cl)
    i1 = -1
    for j in range(ln):
        if c == Cl[j]:
            if n == Nl[j]:
                i1 = j
                break
    if i1 == -1:
        print('Something went wrong in fini', c, Cl, n, Nl)
    
    return i1

def findlowhigh(c, cl): # find the adjacent c values on the c grid, and return their values. 
    a = 0
    while cl[a] < c:
        a += 1
        if a == len(cl):
            print('Something went wrong in findlowhigh', cl[-1], c)
            break
    return cl[a-1], cl[a]
    
def varBV(AR, NB, BV): 
    #Read in the results of the text file: 
    F = open('bv_varnonvar2aboundaries_nrep10000_all.txt')
    Fl = F.readlines()
    Cpbs, Nbs, S1s, S2s, S3s = [], [], [], [], []
    for i in range(1, len(Fl)):
        Cpbs.append(float(Fl[i].split()[0]))
        Nbs.append(float(Fl[i].split()[1]))
        S1s.append(float(Fl[i].split()[2]))
        S2s.append(float(Fl[i].split()[3]))
        S3s.append(float(Fl[i].split()[4]))
    F.close()
    
    n = [50, 135, 370, 1000]
    sc = [0.04, 0.12, 0.4, 1.2, 4.0, 12.0, 40.0, 120.0, 400.0, 1200.0]
    Ns = len(Nbs)
    N = len(AR)
    AR = [40*AR[i] for i in range(N)] # The multiplication of 40 is necessary because the simulations were performed in performed for counts, rather than count rates. 
    VS = [0]*N
    for i in range(N):
        #Determine the 4 corners within which this datapoint is located. 
        #first find whether it is outside the region covered by n and sc. 
        if AR[i] < sc[0]:
            if NB[i] < n[0]:
                #It is in the corner 
                i1 = findi(sc[0], n[0], Cpbs, Nbs)
                qc1, qc2, qc3 = S1s[i1], S2s[i1], S3s[i1]
            elif NB[i] > n[-1]:
                #It is in the other corner
                i1 = findi(sc[0], n[-1], Cpbs, Nbs)
                qc1, qc2, qc3 = S1s[i1], S2s[i1], S3s[i1]
            else:
                #It is somewhere in the range of values for n. 
                na1, na2 = findlowhigh(NB[i], n)
                i1, i2 = findi(sc[0], na1, Cpbs, Nbs), findi(sc[0], na2, Cpbs, Nbs)
                qc1 = 10**(np.log10(S1s[i1]) + ((np.log10(S1s[i2])-np.log10(S1s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                qc2 = 10**(np.log10(S2s[i1]) + ((np.log10(S2s[i2])-np.log10(S2s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                qc3 = 10**(np.log10(S3s[i1]) + ((np.log10(S3s[i2])-np.log10(S3s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                
        elif AR[i] > sc[-1]:
            if NB[i] < n[0]:
                #It is in the corner
                i1 = findi(sc[-1], n[0], Cpbs, Nbs)
                qc1, qc2, qc3 = S1s[i1], S2s[i1], S3s[i1]
            elif NB[i] > n[-1]:
                #It is in the other corner 
                i1 = findi(sc[-1], n[-1], Cpbs, Nbs)
                qc1, qc2, qc3 = S1s[i1], S2s[i1], S3s[i1]
            else:
                #It is somewhere in the range of values for n. 
                na1, na2 = findlowhigh(NB[i], n)
                i1, i2 = findi(sc[-1], na1, Cpbs, Nbs), findi(sc[-1], na2, Cpbs, Nbs)
                qc1 = 10**(np.log10(S1s[i1]) + ((np.log10(S1s[i2])-np.log10(S1s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                qc2 = 10**(np.log10(S2s[i1]) + ((np.log10(S2s[i2])-np.log10(S2s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                qc3 = 10**(np.log10(S3s[i1]) + ((np.log10(S3s[i2])-np.log10(S3s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                
        else:
            if NB[i] < n[0]:
                #It is outside of the range of n, but inside the range of s.
                ra1, ra2 = findlowhigh(AR[i], sc)
                i1, i2 = findi(ra1, n[0], Cpbs, Nbs), findi(ra2, n[0], Cpbs, Nbs)
                qc1 = 10**(np.log10(S1s[i1]) + ((np.log10(S1s[i2])-np.log10(S1s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))
                qc2 = 10**(np.log10(S2s[i1]) + ((np.log10(S2s[i2])-np.log10(S2s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))
                qc3 = 10**(np.log10(S3s[i1]) + ((np.log10(S3s[i2])-np.log10(S3s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))
            
            elif NB[i] > n[-1]:
                #It is outside of the range of n, but inside the range of s.
                ra1, ra2 = findlowhigh(AR[i], sc)
                i1, i2 = findi(ra1, n[-1], Cpbs, Nbs), findi(ra2, n[-1], Cpbs, Nbs)
                qc1 = 10**(np.log10(S1s[i1]) + ((np.log10(S1s[i2])-np.log10(S1s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))
                qc2 = 10**(np.log10(S2s[i1]) + ((np.log10(S2s[i2])-np.log10(S2s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))
                qc3 = 10**(np.log10(S3s[i1]) + ((np.log10(S3s[i2])-np.log10(S3s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))

            else:
                na1, na2 = findlowhigh(NB[i], n)
                ra1, ra2 = findlowhigh(AR[i], sc)                
                #Find the indices of the four corners, and label them 1, 2, 3, 4:
                i1, i2, i3, i4 = findi(ra1, na1, Cpbs, Nbs), findi(ra2, na1, Cpbs, Nbs), findi(ra1, na2, Cpbs, Nbs), findi(ra2, na2, Cpbs, Nbs)
                
                lqc01_1 = np.log10(S1s[i1]) + ((np.log10(S1s[i2]) - np.log10(S1s[i1])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                lqc02_1 = np.log10(S1s[i3]) + ((np.log10(S1s[i4]) - np.log10(S1s[i3])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                qc1 = 10**(lqc01_1 + ((lqc02_1 - lqc01_1) * (np.log10(NB[i]) - np.log10(na1)) / (np.log10(na2) - np.log10(na1))))

                lqc01_2 = np.log10(S2s[i1]) + ((np.log10(S2s[i2]) - np.log10(S2s[i1])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                lqc02_2 = np.log10(S2s[i3]) + ((np.log10(S2s[i4]) - np.log10(S2s[i3])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                qc2 = 10**(lqc01_2 + ((lqc02_2 - lqc01_2) * (np.log10(NB[i]) - np.log10(na1)) / (np.log10(na2) - np.log10(na1))))

                lqc01_3 = np.log10(S3s[i1]) + ((np.log10(S3s[i2]) - np.log10(S3s[i1])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                lqc02_3 = np.log10(S3s[i3]) + ((np.log10(S3s[i4]) - np.log10(S3s[i3])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                qc3 = 10**(lqc01_3 + ((lqc02_3 - lqc01_3) * (np.log10(NB[i]) - np.log10(na1)) / (np.log10(na2) - np.log10(na1))))

        #Now evaluate which variability category this source is: 
        if (qc1 + 4) * (qc2 + 4) * (qc3 + 4) > 0:
            if qc3 > qc2 > qc1:
                if BV[i] > qc3:
                    VS[i] = 3
                elif BV[i] > qc2:
                    VS[i] = 2
                elif BV[i] > qc1:
                    VS[i] = 1
            else:
                print('Something went wrong with the code, BV levels are incorrectly assigned, not increasing', qc1, qc2, qc3)
                
    return VS


def varMAD(AR, NB, MAD): 
    #Read in the results of the text file: 
    F = open('mad_varnonvar2aboundaries_nrep10000_all.txt')
    Fl = F.readlines()
    Cpbs, Nbs, S1s, S2s, S3s = [], [], [], [], []
    for i in range(1, len(Fl)):
        Cpbs.append(float(Fl[i].split()[0]))
        Nbs.append(float(Fl[i].split()[1]))
        S1s.append(float(Fl[i].split()[2]))
        S2s.append(float(Fl[i].split()[3]))
        S3s.append(float(Fl[i].split()[4]))
    F.close()
    
    n = [50, 135, 370, 1000]
    sc = [0.04, 0.12, 0.4, 1.2, 4.0, 12.0, 40.0, 120.0, 400.0, 1200.0]
    Ns = len(Nbs)
    
    #Do this for the list of sources.
    N = len(AR)
    AR = [40*AR[i] for i in range(N)]
    VS = [0]*N
    for i in range(N):
        if AR[i] < sc[0]:
            if NB[i] < n[0]:
                #It is in the corner
                i1 = findi(sc[0], n[0], Cpbs, Nbs)
                qc1, qc2, qc3 = S1s[i1], S2s[i1], S3s[i1]
            elif NB[i] > n[-1]:
                #It is in the other corner
                i1 = findi(sc[0], n[-1], Cpbs, Nbs)
                qc1, qc2, qc3 = S1s[i1], S2s[i1], S3s[i1]
            else:
                #It is somewhere in the range of values for n. 
                na1, na2 = findlowhigh(NB[i], n)
                i1, i2 = findi(sc[0], na1, Cpbs, Nbs), findi(sc[0], na2, Cpbs, Nbs)
                qc1 = 10**(np.log10(S1s[i1]) + ((np.log10(S1s[i2])-np.log10(S1s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                qc2 = 10**(np.log10(S2s[i1]) + ((np.log10(S2s[i2])-np.log10(S2s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                qc3 = 10**(np.log10(S3s[i1]) + ((np.log10(S3s[i2])-np.log10(S3s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                
        elif AR[i] > sc[-1]:
            if NB[i] < n[0]:
                #It is in the corner
                i1 = findi(sc[-1], n[0], Cpbs, Nbs)
                qc1, qc2, qc3 = S1s[i1], S2s[i1], S3s[i1]
            elif NB[i] > n[-1]:
                #It is in the other corner 
                i1 = findi(sc[-1], n[-1], Cpbs, Nbs)
                qc1, qc2, qc3 = S1s[i1], S2s[i1], S3s[i1]
            else:
                #It is somewhere in the range of values for n. 
                na1, na2 = findlowhigh(NB[i], n)
                i1, i2 = findi(sc[-1], na1, Cpbs, Nbs), findi(sc[-1], na2, Cpbs, Nbs)
                qc1 = 10**(np.log10(S1s[i1]) + ((np.log10(S1s[i2])-np.log10(S1s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                qc2 = 10**(np.log10(S2s[i1]) + ((np.log10(S2s[i2])-np.log10(S2s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                qc3 = 10**(np.log10(S3s[i1]) + ((np.log10(S3s[i2])-np.log10(S3s[i1]))*(np.log10(NB[i])-np.log10(na1))/(np.log10(na2)-np.log10(na1))))
                
        else:
            if NB[i] < n[0]:
                #It is outside of the range of n, but inside the range of s. 
                ra1, ra2 = findlowhigh(AR[i], sc)
                i1, i2 = findi(ra1, n[0], Cpbs, Nbs), findi(ra2, n[0], Cpbs, Nbs)
                qc1 = 10**(np.log10(S1s[i1]) + ((np.log10(S1s[i2])-np.log10(S1s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))
                qc2 = 10**(np.log10(S2s[i1]) + ((np.log10(S2s[i2])-np.log10(S2s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))
                qc3 = 10**(np.log10(S3s[i1]) + ((np.log10(S3s[i2])-np.log10(S3s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))
            
            elif NB[i] > n[-1]:
                #It is outside of the range of n, but inside the range of s. 
                ra1, ra2 = findlowhigh(AR[i], sc)
                i1, i2 = findi(ra1, n[-1], Cpbs, Nbs), findi(ra2, n[-1], Cpbs, Nbs)
                qc1 = 10**(np.log10(S1s[i1]) + ((np.log10(S1s[i2])-np.log10(S1s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))
                qc2 = 10**(np.log10(S2s[i1]) + ((np.log10(S2s[i2])-np.log10(S2s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))
                qc3 = 10**(np.log10(S3s[i1]) + ((np.log10(S3s[i2])-np.log10(S3s[i1]))*(np.log10(AR[i])-np.log10(ra1))/(np.log10(ra2)-np.log10(ra1))))

            else:
                na1, na2 = findlowhigh(NB[i], n)
                ra1, ra2 = findlowhigh(AR[i], sc)
                
                #Now find the indices of the four corners, and label them 1, 2, 3, 4:
                i1, i2, i3, i4 = findi(ra1, na1, Cpbs, Nbs), findi(ra2, na1, Cpbs, Nbs), findi(ra1, na2, Cpbs, Nbs), findi(ra2, na2, Cpbs, Nbs)
                
                lqc01_1 = np.log10(S1s[i1]) + ((np.log10(S1s[i2]) - np.log10(S1s[i1])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                lqc02_1 = np.log10(S1s[i3]) + ((np.log10(S1s[i4]) - np.log10(S1s[i3])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                qc1 = 10**(lqc01_1 + ((lqc02_1 - lqc01_1) * (np.log10(NB[i]) - np.log10(na1)) / (np.log10(na2) - np.log10(na1))))

                lqc01_2 = np.log10(S2s[i1]) + ((np.log10(S2s[i2]) - np.log10(S2s[i1])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                lqc02_2 = np.log10(S2s[i3]) + ((np.log10(S2s[i4]) - np.log10(S2s[i3])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                qc2 = 10**(lqc01_2 + ((lqc02_2 - lqc01_2) * (np.log10(NB[i]) - np.log10(na1)) / (np.log10(na2) - np.log10(na1))))

                lqc01_3 = np.log10(S3s[i1]) + ((np.log10(S3s[i2]) - np.log10(S3s[i1])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                lqc02_3 = np.log10(S3s[i3]) + ((np.log10(S3s[i4]) - np.log10(S3s[i3])) * (np.log10(AR[i]) - np.log10(ra1)) / (np.log10(ra2) - np.log10(ra1)))
                qc3 = 10**(lqc01_3 + ((lqc02_3 - lqc01_3) * (np.log10(NB[i]) - np.log10(na1)) / (np.log10(na2) - np.log10(na1))))

        #Now evaluate which variability category this source is: 
        if (qc1 + 4) * (qc2 + 4) * (qc3 + 4) > 0:
            if qc3 > qc2 > qc1:
                if MAD[i] > qc3:
                    VS[i] = 3
                elif MAD[i] > qc2:
                    VS[i] = 2
                elif MAD[i] > qc1:
                    VS[i] = 1
            else:
                print('Something went wrong with the code, MAD levels are incorrectly assigned, not increasing', qc1, qc2, qc3)
                
    return VS

MAD = [0]*Nlc
bv10 = [0]*Nlc
AR, ARr = [0]*Nlc, [0]*Nlc
nb, nbr = [0]*Nlc, [0]*Nlc

#Open the files, generate the light curves, and then distinguish them as variable/ not variable.

for i in range(Nlc):
    #check to see if light curve exists.
    if os.path.isfile(lcf[i]) == 0:
        print(lcf[i], 'DOES NOT EXIST!')
        continue
    print('\n\nWorking on source', i, '/ ', Nlc)
    ifl = fits.open(lcf[i])
    T0 = ifl[1].data["Time"]
    C0 = ifl[1].data["Counts"]
    BC0 = ifl[1].data["Back_counts"]
    FE0 = ifl[1].data["Fracexp"]
    BA0 = ifl[1].data["backratio"]
    ifl.close()
    n = len(T0)

    #Assign these data to distinct eroDays:
    ise_erd = [[0]]
    j = 0
    for k in range(n):
        if T0[k]-T0[ise_erd[j][0]] > 2*60*60:
            ise_erd[j].append(k)
            ise_erd.append([k])
            j += 1
    Ned = len(ise_erd)
    ise_erd[Ned-1].append(n)

    #Create new arrays that have exactly one bin per eroday. Only take the one energy band selected. 
    C = [0]*Ned
    BC = [0]*Ned
    FE = [0]*Ned
    BA = [0]*Ned

    for l in range(Ned):
        for k in range(ise_erd[l][0], ise_erd[l][1]):
            if np.isinf(C0[k][eb]) + np.isnan(C0[k][eb]) + np.isinf(BC0[k][eb]) + np.isnan(BC0[k][eb]) + np.isinf(FE0[k][eb]) + np.isnan(FE0[k][eb]) + np.isinf(BA0[k]) + np.isnan(BA0[k]) == 0:
                C[l] += C0[k][eb]
                BC[l] += BC0[k][eb]
                FE[l] += FE0[k][eb]
                BA[l] += BA0[k] * FE0[k][eb]
        if FE[l] > 0:
            BA[l] /= FE[l]

    #Delete bins that have a too small FE.
    k = 0
    while k < len(FE):
        if FE[k] < minFE:
            del C[k], BC[k], FE[k], BA[k]
        else:
            k += 1

    #Very low count rate sources need to be rebinned for SCATT_LO to work properly
    AC = (np.array(C).sum() - np.array([BC[k] * BA[k] for k in range(len(C))]).sum()) / Ned # determines average source counts per eroday.
    if AC <= 0:
        print('\nLight curve', lcf[i], 'has a too low count rate:', AC)
        rbf = 1
        Cr = C.copy()
        BCr = BC.copy()
        FEr = FE.copy()
        BAr = BA.copy()
    elif AC < 0.5:
        rbf = math.ceil(0.5/AC)
        if math.ceil(len(C)/rbf) < 20:
            rbf = math.floor(len(C)/20) # Should have at least 20 bins left for analysis.
        ise_erdr = [[0]]
        j = 0
        for k in range(n):
            if T0[k]-T0[ise_erdr[j][0]] > (4*rbf - 2)*60*60:
                ise_erdr[j].append(k)
                ise_erdr.append([k])
                j+=1

        Nedr = len(ise_erdr)
        ise_erdr[Nedr-1].append(n)

        Cr = [0]*Nedr
        BCr = [0]*Nedr
        FEr = [0]*Nedr
        BAr = [0]*Nedr

        for l in range(Nedr):
            for k in range(ise_erdr[l][0], ise_erdr[l][1]):
                if np.isinf(C0[k][m]) + np.isnan(C0[k][m]) + np.isinf(BC0[k][m]) + np.isnan(BC0[k][m]) + np.isinf(FE0[k][m]) + np.isnan(FE0[k][m]) + np.isinf(BA0[k]) + np.isnan(BA0[k]) == 0:
                    C[l] += C0[k][eb]
                    BC[l] += BC0[k][eb]
                    FE[l] += FE0[k][eb]
                    BA[l] += BA0[k] * FE0[k][eb]
            if FE[l] > 0:
                BA[l] /= FE[l]

        #Delete bins that have a too small FE.
        k = 0
        while k < len(FEr):
            if FEr[k] < minFE:
                del Cr[k], BCr[k], FEr[k], BAr[k]
            else:
                k += 1

    else:
        rbf = 1
        Cr = C.copy()
        BCr = BC.copy()
        FEr = FE.copy()
        BAr = BA.copy()

    #Determine the AMPL_SIG value based on the unrebinned dataset:
    if len(C) == 0:
        print('All datapoints below minFE')
    else:
        R, Rem, Rep = errbar(np.array(C), BC, BA, FE, n_gp, dted)
        Rpe, Rme = R+Rep, R-Rem
        miRpe = min(Rpe) #min(R+e)
        maRme = max(Rme) #max(R-e)
        smiR = Rep[np.where(Rpe == min(Rpe))][0]
        smaR = Rem[np.where(Rme == max(Rme))][0]
        MAD[i] = (maRme - miRpe) / np.sqrt(smiR**2 + smaR**2)
        AR[i] = (np.array(C).sum() - np.array([BC[k] * BA[k] for k in range(len(BC))]).sum())/(np.array(FE).sum()*dted)
        nb[i] = len(C)

    #Determine the SCATT_LO value based on the rebinned dataset:
    if len(Cr) == 0:
        print('All datapoints below minFE')
    else:
        lm, ls = run_bexvar(np.array(Cr), np.array(BCr), np.array(BAr), np.array(FEr), n_gp)
        bs = sorted(ls)
        bv10[i] = bs[int(math.floor(0.1*(len(bs)-1)))]
        ARr[i] = (np.array(Cr).sum() - np.array([BCr[k] * BAr[k] for k in range(len(BCr))]).sum())/(np.array(FEr).sum()*dted)
        nbr[i] = len(Cr)

#Now identify whether these light curves are variable
MVC = varMAD(AR, nb, MAD)
BVC = varBV(AR, nbr, bv10)
VC = [0]*Nlc

for i in range(Nlc):
    if MVC[i] == 3:
        VC[i] = 1
    if BVC[i] == 3:
        VC[i] = 1

print('\nVariability classification:')
print('Source\tSCATT_LO var class \tAMPL_SIG var class\tTotal var class')
for i in range(Nlc):
    print(lcf[i] + '\t' + str(BVC[i]) + '\t' + str(MVC[i]) + '\t' + str(VC[i]))

#Write the results to file:
if os.path.isfile('Identified_variable_sources.fits') == 1:
    os.remove('Identified_variable_sources.fits')

t = Table([lcf, BVC, MVC, VC], names=('Light_curve_file', 'SCATT_LO_varclass', 'AMPL_SIG_varclass', 'Varclass'))
t.write('Identified_variable_sources.fits', format='fits')
