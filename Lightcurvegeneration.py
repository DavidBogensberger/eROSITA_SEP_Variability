#This script generates a set of graphs of light curves spanning multiple eRASSs. It separates the data into distinct segments of observations, which may or may not match the eRASS distinctions. It creates both non-rebinned, and rebinned graphs.
#For this code to work, the light curves it uses should have been created by srctool with the following parameters:
#todo="LC LCCORR" insts="1 2 3 4 5 6 7" srcreg="AUTO" backreg="AUTO" lctype="REGULAR-" lcpars="40.0" lcemin="0.2 0.6 2.3 0.2" lcemax="0.6 2.3 5.0 5.0" lcgamma="1.9"

from errorbar_calc1 import *
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib
import math
import os.path

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
plt.rcParams.update({'font.size': 17})

#Select this energy band
ebmm = "0.2-5.0"
m = 3 #Corresponds to "0.2-5.0"
minFE = 0.1 #minimum fracexp

qum = [0.15865, 0.84135] #1 sigma quantiles used for error estimation

dted = 40 #The length of an eroday observation.
n_gp = 300 #number of gridpoints for error estimation

color= ['#1b9e77', '#d95f02', '#7570b3'] #The colours used for eRASS1, 2, and 3 in the graphs

#Load in the list of stl, lcn of the matched variable sources:
F = fits.open('FVS2d_varsrc_final_all_edit3.fits')
stl1 = F[1].data['Skytile_v1']
lcn1 = F[1].data['Light_Curve_Number_v1']
stl2 = F[1].data['Skytile_v2']
lcn2 = F[1].data['Light_Curve_Number_v2']
stl3 = F[1].data['Skytile_v3']
lcn3 = F[1].data['Light_Curve_Number_v3']
RA1 = F[1].data['RA_v1']
Dec1 = F[1].data['Dec_v1']
RA2 = F[1].data['RA_v2']
Dec2 = F[1].data['Dec_v2']
RA3 = F[1].data['RA_v3']
Dec3 = F[1].data['Dec_v3']
RAx = F[1].data['RA_c947']
Decx = F[1].data['Dec_c947']
ev1 = F[1].data['var_cat_v1']
ev2 = F[1].data['var_cat_v2']
ev3 = F[1].data['var_cat_v3']
F.close()

nv = len(stl1)

#Select which sources to run the code on. In this example, it only runs for source 0 in the catalogue. 
for i in range(0,1):
    R, Rem, Rep, T, Ca, BCa, BAa, FEa = [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)]
    Rr, Remr, Repr, Tr, Cr, BCr, BAr, FEr = [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)], [[] for r in range(3)]
    #Define the total properties as well:
    Tt, Rt, Remt, Rept, Ttr, Rtr, Remtr, Reptr, eRn, eRnr = [], [], [], [], [], [], [], [], [], []
    rbfr = [0]*3 # saving the rbf, to be used later. 
    for r in range(1, 4):
        print('\n\nWorking on source', i, '/', nv, ', and eRASS:', r, '\n\n')
        if vars()['stl'+str(r)][i] != '-1':
            #Check if file exists.
            #Put path to file here.
            if os.path.isfile('em0'+str(r)+'_'+vars()['stl'+str(r)][i]+'_020_020_LightCurve_'+vars()['lcn'+str(r)][i]+'_c946.fits') == 0:
                print('\nem0'+str(r)+'_'+vars()['stl'+str(r)][i]+'_020_020_LightCurve_'+vars()['lcn'+str(r)][i]+'_c946.fits DOES NOT EXIST!\n')
                continue
            ifl = fits.open('em0'+str(r)+'_'+vars()['stl'+str(r)][i]+'_020_020_LightCurve_'+vars()['lcn'+str(r)][i]+'_c946.fits')
            Tmjdref = ifl[1].header["MJDREF"]
            dt = ifl[1].header["TIMEDEL"]
            T0 = ifl[1].data["Time"]
            C0 = ifl[1].data["Counts"]
            BC0 = ifl[1].data["Back_counts"]
            FE0 = ifl[1].data["Fracexp"]
            BA0 = ifl[1].data["backratio"]
            ifl.close()
            n = len(T0)

            Tmjd = [Tmjdref+(T0[j]/(24*60*60))-58800 for j in range(n)] #use this instead, to get more manageable x axes. 

            #The following is to check whether there are any instances where more than one bin of 40s is required to cover one eroday.
            ise_erd = [[0]]
            k = 0
            for l in range(n):
                if T0[l]-T0[ise_erd[k][0]] > 2*60*60:
                    ise_erd[k].append(l)
                    ise_erd.append([l])
                    k+=1
            Ned = len(ise_erd)
            ise_erd[Ned-1].append(n)

            T_0 = [0]*Ned
            C = [0]*Ned
            BC = [0]*Ned
            FE = [0]*Ned
            BA = [0]*Ned

            for l in range(Ned):
                nbped = 0
                for k in range(ise_erd[l][0], ise_erd[l][1]):
                    if FE0[k][m] > 0:
                        if np.isinf(C0[k][m]) + np.isnan(C0[k][m]) + np.isinf(BC0[k][m]) + np.isnan(BC0[k][m]) + np.isinf(FE0[k][m]) + np.isnan(FE0[k][m]) + np.isinf(BA0[k]) + np.isnan(BA0[k])  == 0:
                            C[l] += C0[k][m]
                            BC[l] += BC0[k][m]
                            FE[l] += FE0[k][m]
                            BA[l] += BA0[k]
                            T_0[l] += Tmjd[k] * FE0[k][m]
                            nbped += 1
                if nbped > 0:
                    BA[l] /= nbped
                if FE[l] > 0:
                    T_0[l] /= FE[l]

            #Now delete the bins that have a too small FE.
            u = 0
            while u < len(FE):
                if FE[u] < minFE:
                    del C[u], BC[u], FE[u], BA[u], T_0[u]
                else:
                    u += 1

            #Now determine the count rates and error bars:
            R[r-1], Rem[r-1], Rep[r-1] = errbar(C, BC, BA, FE, n_gp, dted)
            T[r-1] = T_0.copy()
            Ca[r-1] = C.copy()
            BCa[r-1] = BC.copy()
            BAa[r-1] = BA.copy()
            FEa[r-1] = FE.copy()

            #Now for the rebinning part
            AC = 0
            for q in range(len(C0)):
                if np.isnan(C0[q][m]) + np.isinf(C0[q][m]) + np.isnan(BC0[q][m]) + np.isinf(BC0[q][m]) + np.isnan(BA0[q]) + np.isinf(BA0[q]) == 0:
                    AC += C0[q][m] - (BC0[q][m] * BA0[q])
            AC /= Ned
            print('AC:', AC)
            if AC <= 0:
                print('Source', vars()['stl'+str(r)][i], vars()['lcn'+str(r)][i], 'has a too low count rate:', AC)
                rbf = 2 # minimum rebin factor is 2. 
                rbfr[r-1] = 2
            if AC < 100000: #Arbitrary large value. 
                if AC > 0:
                    rbf = math.ceil(20.0/AC)
                    rbfr[r-1] = math.ceil(20.0/AC)
                else:
                    rbf = 100000
                    rbfr[r-1] = 100000
                if math.ceil(Ned/rbf) < 20:
                    rbf = math.floor(Ned/20) #Maximum bin depth should leave 20 bins left for analysis.
                    rbfr[r-1] = math.floor(Ned/20)

                #Want a rebin factor of at least 2.
                if rbf == 1:
                    rbf = 2
                    rbfr[r-1] = 2

                #Group data into the rebinned bins
                ise_erdr = [[0]]
                j = 0
                for k in range(n):
                    if T0[k]-T0[ise_erdr[j][0]] > (4*rbf - 2)*60*60:
                        ise_erdr[j].append(k)
                        ise_erdr.append([k])
                        j+=1

                Nedr = len(ise_erdr)
                ise_erdr[Nedr-1].append(n)

                T_0 = [0]*Nedr
                C = [0]*Nedr
                BC = [0]*Nedr
                FE = [0]*Nedr
                BA = [0]*Nedr

                for l in range(Nedr):
                    nbped = 0
                    for k in range(ise_erdr[l][0], ise_erdr[l][1]):
                        if FE0[k][m] > 0:
                            if np.isinf(C0[k][m]) + np.isnan(C0[k][m]) + np.isinf(BC0[k][m]) + np.isnan(BC0[k][m]) + np.isinf(FE0[k][m]) + np.isnan(FE0[k][m]) + np.isinf(BA0[k]) + np.isnan(BA0[k])  == 0:
                                C[l] += C0[k][m]
                                BC[l] += BC0[k][m]
                                FE[l] += FE0[k][m] / rbf
                                BA[l] += BA0[k]
                                T_0[l] += Tmjd[k] * FE0[k][m] / rbf # the dividing by rbf is to make the scaling later easier. 
                                nbped += 1
                    if nbped > 0:
                        BA[l] /= nbped
                    if FE[l] > 0:
                        T_0[l] /= FE[l]

                #Now delete the bins that have a too small FE.
                u = 0
                while u < len(FE):
                    if FE[u] < minFE:
                        del C[u], BC[u], FE[u], BA[u], T_0[u]
                    else:
                        u += 1

                #Now determine the count rates and error bars:
                #print(C, BC, BA, FE, n_gp, dted)
                if len(C) > 0:
                    #print(C[m], BC[m], BA[m], FE[m], n_gp, dted*rbf)
                    Rr[r-1], Remr[r-1], Repr[r-1] = errbar(C, BC, BA, FE, n_gp, dted*rbf)
                    Tr[r-1] = T_0.copy()
                    Cr[r-1] = C.copy()
                    BCr[r-1] = BC.copy()
                    BAr[r-1] = BA.copy()
                    FEr[r-1] = FE.copy()

            else:
                Tr[r-1], Rr[r-1], Remr[r-1], Repr[r-1], Cr[r-1], BCr[r-1], BAr[r-1], FEr[r-1] = T[r-1].copy(), R[r-1].copy(), Rem[r-1].copy(), Rep[r-1].copy(), Ca[r-1].copy(), BCa[r-1].copy(), BAa[r-1].copy(), FEa[r-1].copy()
                rbfr[r-1] = 1

                        
    #And now plot the graph for them:
    #Determine which RA, Dec to quote.
    if RAx[i] != -1:
        RA, Dec = RAx[i], Decx[i]
    else:
        if RA1[i] != -1:
            RA, Dec = RA1[i], Dec1[i]
        else:
            if RA2[i] != -1:
                RA, Dec = RA2[i], Dec2[i]
            else:
                if RA3[i] != -1:
                    RA, Dec = RA3[i], Dec3[i]

    #Combine the dataset into the total dataset:
    for r in range(1, 4):
        for j in range(len(T[r-1])):
            Tt.append(T[r-1][j])
            Rt.append(R[r-1][j])
            Remt.append(Rem[r-1][j])
            Rept.append(Rep[r-1][j])
            eRn.append(r-1)
        for j in range(len(Tr[r-1])):
            Ttr.append(Tr[r-1][j])
            Rtr.append(Rr[r-1][j])
            Remtr.append(Remr[r-1][j])
            Reptr.append(Repr[r-1][j])
            eRnr.append(r-1)
            
    #Now split into segments.
    ttg = [] # the time of the gaps. To be used when determining the bins for the rebinned light curve.
    ns = 1
    js = [[0]]
    for j in range(1, len(Tt)):
        if Tt[j]-Tt[j-1] > 5: #Make this the limiting condition
            ns += 1
            js[-1].append(j)
            js.append([j])
            ttg.append((Tt[j]+Tt[j-1])/2)
    js[-1].append(len(Tt)-1)

    #Make the final entry of ttg very, very large:
    ttg.append(1000000)
    #Delete segments that have too few bins:
    k = 0
    while k < len(js):
        if js[k][1] - js[k][0] < 5: #Make this the limiting condition; at least 5 bins per segment.
            del js[k]
            ns -= 1
        else:
            k += 1

    print('Number of segments:', ns, js)

    mrbf = max(rbfr)  #taking the max rbf for determining the necessary gaps between data. 

    nsr = 1
    jsr = [[0]]

    #Instead, use the ttg:
    ttgi = 0
    for j in range(1, len(Ttr)):
        if Ttr[j] > ttg[ttgi]:
            ttgi += 1
            nsr += 1
            jsr[-1].append(j)
            jsr.append([j])
    jsr[-1].append(len(Ttr)-1)
    #Delete segments that have too few bins:
    k = 0
    while k < len(jsr):
        if jsr[k][1] - jsr[k][0] < 3: #Make this the limiting condition; at least 3 bins per segment for the rebinned light curve. 
            del jsr[k]
            nsr -= 1
        else:
            k += 1

    print('Number of segments, rebinned lightcurve:', nsr, jsr)

    #Split the data into these segments: 
    Ts, Rs, Reps, Rems, eRns = [[] for j in range(ns)], [[] for j in range(ns)], [[] for j in range(ns)], [[] for j in range(ns)], [[] for j in range(ns)]
    Tsr, Rsr, Repsr, Remsr, eRnsr = [[] for j in range(nsr)], [[] for j in range(nsr)], [[] for j in range(nsr)], [[] for j in range(nsr)], [[] for j in range(nsr)]

    for j in range(ns):
        for k in range(js[j][0], js[j][1]):
            Ts[j].append(Tt[k])
            Rs[j].append(Rt[k])
            Reps[j].append(Rept[k])
            Rems[j].append(Remt[k])
            eRns[j].append(eRn[k])
    for j in range(nsr):
        for k in range(jsr[j][0], jsr[j][1]):
            Tsr[j].append(Ttr[k])
            Rsr[j].append(Rtr[k])
            Repsr[j].append(Reptr[k])
            Remsr[j].append(Remtr[k])
            eRnsr[j].append(eRnr[k])

    #Using the eRns, and eRnsr, split all the segments into eRASS1, 2, 3, so that I can colour them accordingly.
    Tser, Rser, Repser, Remser = [[[] for r in range(3)] for j in range(ns)], [[[] for r in range(3)] for j in range(ns)], [[[] for r in range(3)] for j in range(ns)], [[[] for r in range(3)] for j in range(ns)]
    Tsrer, Rsrer, Repsrer, Remsrer = [[[] for r in range(3)] for j in range(nsr)], [[[] for r in range(3)] for j in range(nsr)], [[[] for r in range(3)] for j in range(nsr)], [[[] for r in range(3)] for j in range(nsr)]
    for j in range(ns):
        for k in range(js[j][0], js[j][1]):
            Tser[j][eRn[k]].append(Tt[k])
            Rser[j][eRn[k]].append(Rt[k])
            Repser[j][eRn[k]].append(Rept[k])
            Remser[j][eRn[k]].append(Remt[k])
    for j in range(nsr):
        for k in range(jsr[j][0], jsr[j][1]):
            Tsrer[j][eRnr[k]].append(Ttr[k])
            Rsrer[j][eRnr[k]].append(Rtr[k])
            Repsrer[j][eRnr[k]].append(Reptr[k])
            Remsrer[j][eRnr[k]].append(Remtr[k])
    

    #Make the plot that distinguishes the data into segments, and plot them in adjacent figures.
    fm0 = [min([Rt[j] - Remt[j] for j in range(len(Rt))]), max([Rt[j] + Rept[j] for j in range(len(Rt))])]
    ym = [fm0[0] - 0.02*(fm0[1]-fm0[0]), fm0[1] + 0.02*(fm0[1]-fm0[0])]
    #Also determine the x range of each of the segments:
    xm = [[Ts[j][0] - 2, Ts[j][-1] + 2] for j in range(ns)]
    wr = [xm[j][1]-xm[j][0] for j in range(ns)]

    #Make a version here that works for ns == 1:
    if ns == 1:

        fig, ax = plt.subplots(1, 1, figsize=(10,6))
        for r in range(3):
            if Tser[0][r] != []:
                ax.errorbar(Tser[0][r], Rser[0][r], yerr=[Remser[0][r], Repser[0][r]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black')
        ax.set_xlim(xm[0][0], xm[0][1])
        ax.set_ylim(ym[0], ym[1])
        ax.tick_params(length=7, width=1, which='major', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
        ax.tick_params(length=3.5, width=1, which='minor', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2.0)
        #for the legend:
        for r in range(3):
            if vars()['stl'+str(r+1)][i] != '-1':
                ax.errorbar([0, 0], [0, 0], yerr=[[1, 1], [1, 1]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black', label='eRASS'+str(r+1))
        plt.legend(prop={'size':13})
        plt.subplots_adjust(bottom=0.15)
        namt = 'Light curve for variable source at '+str(round(RA,5))+', '+str(round(Dec,5))+'\nfound to be variable in eRASS:'
        nam = 'LCseg_'+str(round(RA,5))+'_'+str(round(Dec,5))+'_var_e'
        for r in range(1,4):
            if vars()['ev'+str(r)][i] > 0:
                namt += str(r)+', '
                nam += str(r)
        namt += ' with eR, stl, lcn: \n'
        for r in range(1,4):
            namt += str(r)+', '+str(vars()['stl'+str(r)][i])+', '+str(vars()['lcn'+str(r)][i])+', '
        namt += str(ebmm)
        print('Savefig name: ', nam)
        ax.set_ylabel('Count rate (cts/s)')
        ax.set_xlabel('Time (MJD) - 58800')
        plt.savefig(nam+'.png', format='png')
        plt.savefig(nam+'.svg', format='svg', dpi=3200)
        plt.close()
        
    #The following only works if ns > 1:
    if ns > 1:
        #consider even and odd instances:
        if ns%2 == 1: #odd
            #Make a version with no lines:
            fig, ax = plt.subplots(1, ns, sharey=True, gridspec_kw={'width_ratios': wr}, figsize=(10,6))
            fig.subplots_adjust(wspace=0)
            for j in range(ns):
                for r in range(3):
                    if Tser[j][r] != []:
                        ax[j].errorbar(Tser[j][r], Rser[j][r], yerr=[Remser[j][r], Repser[j][r]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black')
                ax[j].set_xlim(xm[j][0], xm[j][1])
                ax[j].set_ylim(ym[0], ym[1])
            ax[0].tick_params(length=7, width=1, which='major', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
            ax[0].tick_params(length=3.5, width=1, which='minor', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[0].spines[axis].set_linewidth(2.0)
            for j in range(1, ns):
                ax[j].tick_params(length=7, width=1, which='major', colors='black', direction='out', bottom=1, top=0, left=0, right=0)
                ax[j].tick_params(length=3.5, width=1, which='minor', colors='black', direction='out', bottom=1, top=0, left=0, right=0)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax[j].spines[axis].set_linewidth(2.0)
            #for the legend:
            for r in range(3):
                if vars()['stl'+str(r+1)][i] != '-1':
                    ax[ns-1].errorbar([0, 0], [0, 0], yerr=[[1, 1], [1, 1]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black', label='eRASS'+str(r+1))
            plt.legend(prop={'size':13})
            plt.subplots_adjust(bottom=0.15)
            namt = 'Light curve for variable source at '+str(round(RA,5))+', '+str(round(Dec,5))+'\nfound to be variable in eRASS:'
            nam = 'LCseg_'+str(round(RA,5))+'_'+str(round(Dec,5))+'_var_e'
            for r in range(1,4):
                if vars()['ev'+str(r)][i] > 0:
                    namt += str(r)+', '
                    nam += str(r)
            namt += ' with eR, stl, lcn: \n'
            for r in range(1,4):
                namt += str(r)+', '+str(vars()['stl'+str(r)][i])+', '+str(vars()['lcn'+str(r)][i])+', '
            namt += str(ebmm)
            print('Savefig name: ', nam)
            #ax[int(math.floor((ns-1)/2))].set_title(namt)
            ax[0].set_ylabel('Count rate (cts/s)')
            ax[int(math.floor((ns-1)/2))].set_xlabel('Time (MJD) - 58800')
            plt.savefig(nam+'.png', format='png')
            plt.savefig(nam+'.svg', format='svg', dpi=3200)
            plt.close()

        else: # if even, then add an additional subfigure that has 0 width.
            #Make a version with no lines:
            wr1 = []
            for _ in range(int(ns/2)):
                wr1.append(wr[_])
            wr1.append(0.0)
            for _ in range(int(ns/2), ns):
                wr1.append(wr[_])
            fig, ax = plt.subplots(1, ns+1, sharey=True, gridspec_kw={'width_ratios': wr1}, figsize=(10,6))
            fig.subplots_adjust(wspace=0)
            for j in range(ns):
                if j >= int(ns/2):
                    j1 = j+1
                else:
                    j1 = j
                for r in range(3):
                    if Tser[j][r] != []:
                        ax[j1].errorbar(Tser[j][r], Rser[j][r], yerr=[Remser[j][r], Repser[j][r]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black')
                ax[j1].set_xlim(xm[j][0], xm[j][1])
                ax[j1].set_ylim(ym[0], ym[1])
            ax[0].tick_params(length=7, width=1, which='major', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
            ax[0].tick_params(length=3.5, width=1, which='minor', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[0].spines[axis].set_linewidth(2.0)
            for j in range(1, ns+1):
                if j == int(ns/2):
                    continue
                ax[j].tick_params(length=7, width=1, which='major', colors='black', direction='out', bottom=1, top=0, left=0, right=0)
                ax[j].tick_params(length=3.5, width=1, which='minor', colors='black', direction='out', bottom=1, top=0, left=0, right=0)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax[j].spines[axis].set_linewidth(2.0)
            #for the legend:
            for r in range(3):
                if vars()['stl'+str(r+1)][i] != '-1':
                    ax[ns].errorbar([0, 0], [0, 0], yerr=[[1, 1], [1, 1]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black', label='eRASS'+str(r+1))
            plt.legend(prop={'size':13})
            plt.subplots_adjust(bottom=0.15)
            namt = 'Light curve for variable source at '+str(round(RA,5))+', '+str(round(Dec,5))+'\nfound to be variable in eRASS:'
            nam = 'LCseg_'+str(round(RA,5))+'_'+str(round(Dec,5))+'_var_e'
            for r in range(1,4):
                if vars()['ev'+str(r)][i] > 0:
                    namt += str(r)+', '
                    nam += str(r)
            namt += ' with eR, stl, lcn: \n'
            for r in range(1,4):
                namt += str(r)+', '+str(vars()['stl'+str(r)][i])+', '+str(vars()['lcn'+str(r)][i])+', '
            namt += str(ebmm)
            print('Savefig name: ', nam)
            ax[0].set_ylabel('Count rate (cts/s)')
            ax[int(ns/2)].set_xlabel('\n\nTime (MJD) - 58800')
            ax[int(ns/2)].axes.xaxis.set_ticks([])
            plt.savefig(nam+'.png', format='png')
            plt.savefig(nam+'.svg', format='svg', dpi=3200)
            plt.close()
            

    #Now repeat the above for the rebinned dataset:
    fm0 = [min([Rtr[j] - Remtr[j] for j in range(len(Rtr))]), max([Rtr[j] + Reptr[j] for j in range(len(Rtr))])]
    ym = [fm0[0] - 0.02*(fm0[1]-fm0[0]), fm0[1] + 0.02*(fm0[1]-fm0[0])]
    #Also determine the x range of each of the segments:
    xm = [[Tsr[j][0] - 2, Tsr[j][-1] + 2] for j in range(nsr)]
    wr = [xm[j][1]-xm[j][0] for j in range(nsr)]
    if nsr == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10,6))
        for r in range(3):
            if Tsrer[0][r] != []:
                ax.errorbar(Tsrer[0][r], Rsrer[0][r], yerr=[Remsrer[0][r], Repsrer[0][r]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black')
        ax.set_xlim(xm[0][0], xm[0][1])
        ax.set_ylim(ym[0], ym[1])
        ax.tick_params(length=7, width=1, which='major', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
        ax.tick_params(length=3.5, width=1, which='minor', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2.0)
        #for the legend:
        for r in range(3):
            if vars()['stl'+str(r+1)][i] != '-1':
                ax.errorbar([0, 0], [0, 0], yerr=[[1, 1], [1, 1]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black', label='eRASS'+str(r+1))
        plt.legend(prop={'size':13})
        plt.subplots_adjust(bottom=0.15)
        namt = 'Light curve for variable source at '+str(round(RA,5))+', '+str(round(Dec,5))+'\nfound to be variable in eRASS:'
        nam = 'LCseg_'+str(round(RA,5))+'_'+str(round(Dec,5))+'_var_e'
        for r in range(1,4):
            if vars()['ev'+str(r)][i] > 0:
                namt += str(r)+', '
                nam += str(r)
        namt += ' with eR, stl, lcn: \n'
        for r in range(1,4):
            namt += str(r)+', '+str(vars()['stl'+str(r)][i])+', '+str(vars()['lcn'+str(r)][i])+', '
        namt += str(ebmm)
        print('Savefig name: ', nam)
        ax.set_ylabel('Count rate (cts/s)')
        ax.set_xlabel('Time (MJD) - 58800')
        plt.savefig(nam+'_rebin.png', format='png')
        plt.savefig(nam+'_rebin.svg', format='svg', dpi=3200)
        plt.close()
        
    if nsr > 1:
        #Make a version with no lines:
        #consider even and odd instances:
        if nsr%2 == 1: #odd
            fig, ax = plt.subplots(1, nsr, sharey=True, gridspec_kw={'width_ratios': wr}, figsize=(10,6))
            fig.subplots_adjust(wspace=0)
            for j in range(nsr):
                for r in range(3):
                    if Tsrer[j][r] != []:
                        ax[j].errorbar(Tsrer[j][r], Rsrer[j][r], yerr=[Remsrer[j][r], Repsrer[j][r]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black')
                ax[j].set_xlim(xm[j][0], xm[j][1])
                ax[j].set_ylim(ym[0], ym[1])
            ax[0].tick_params(length=7, width=1, which='major', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
            ax[0].tick_params(length=3.5, width=1, which='minor', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[0].spines[axis].set_linewidth(2.0)
            for j in range(1, nsr):
                ax[j].tick_params(length=7, width=1, which='major', colors='black', direction='out', bottom=1, top=0, left=0, right=0)
                ax[j].tick_params(length=3.5, width=1, which='minor', colors='black', direction='out', bottom=1, top=0, left=0, right=0)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax[j].spines[axis].set_linewidth(2.0)
            #for the legend:
            for r in range(3):
                if vars()['stl'+str(r+1)][i] != '-1':
                    ax[nsr-1].errorbar([0, 0], [0, 0], yerr=[[1, 1], [1, 1]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black', label='eRASS'+str(r+1))
            plt.legend(prop={'size':13})
            plt.subplots_adjust(bottom=0.15)
            namt = 'Light curve for variable source at '+str(round(RA,5))+', '+str(round(Dec,5))+'\nfound to be variable in eRASS:'
            nam = 'LCseg_'+str(round(RA,5))+'_'+str(round(Dec,5))+'_var_e'
            for r in range(1,4):
                if vars()['ev'+str(r)][i] > 0:
                    namt += str(r)+', '
                    nam += str(r)
            namt += ' with eR, stl, lcn: \n'
            for r in range(1,4):
                namt += str(r)+', '+str(vars()['stl'+str(r)][i])+', '+str(vars()['lcn'+str(r)][i])+', '
            namt += str(ebmm)
            print('Savefig name: ', nam)
            ax[0].set_ylabel('Count rate (cts/s)')
            ax[int(math.floor((nsr-1)/2))].set_xlabel('Time (MJD) - 58800')
            plt.savefig(nam+'_rebin.png', format='png')
            plt.savefig(nam+'_rebin.svg', format='svg', dpi=3200)
            plt.close()

        else: # if nsr is even:
            wr1 = []
            for _ in range(int(nsr/2)):
                wr1.append(wr[_])
            wr1.append(0.0)
            for _ in range(int(nsr/2)):
                wr1.append(wr[int(nsr/2)+_])
            fig, ax = plt.subplots(1, nsr+1, sharey=True, gridspec_kw={'width_ratios': wr1}, figsize=(10,6))
            fig.subplots_adjust(wspace=0)
            for j in range(nsr):
                if j >= int(nsr/2):
                    j1 = j+1
                else:
                    j1 = j
                for r in range(3):
                    if Tsrer[j][r] != []:
                        ax[j1].errorbar(Tsrer[j][r], Rsrer[j][r], yerr=[Remsrer[j][r], Repsrer[j][r]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black')
                ax[j1].set_xlim(xm[j][0], xm[j][1])
                ax[j1].set_ylim(ym[0], ym[1])
            ax[0].tick_params(length=7, width=1, which='major', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
            ax[0].tick_params(length=3.5, width=1, which='minor', colors='black', direction='out', bottom=1, top=0, left=1, right=0)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[0].spines[axis].set_linewidth(2.0)
            for j in range(1, nsr+1):
                if j == int(nsr/2):
                    continue
                ax[j].tick_params(length=7, width=1, which='major', colors='black', direction='out', bottom=1, top=0, left=0, right=0)
                ax[j].tick_params(length=3.5, width=1, which='minor', colors='black', direction='out', bottom=1, top=0, left=0, right=0)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax[j].spines[axis].set_linewidth(2.0)
            #for the legend:
            for r in range(3):
                if vars()['stl'+str(r+1)][i] != '-1':
                    ax[nsr].errorbar([0, 0], [0, 0], yerr=[[1, 1], [1, 1]], fmt='o', elinewidth=0.5, capsize=0, color=color[r], markersize=4.5, markeredgewidth=0.5, markeredgecolor='black', label='eRASS'+str(r+1))
            plt.legend(prop={'size':13})
            plt.subplots_adjust(bottom=0.15)
            namt = 'Light curve for variable source at '+str(round(RA,5))+', '+str(round(Dec,5))+'\nfound to be variable in eRASS:'
            nam = 'LCseg_'+str(round(RA,5))+'_'+str(round(Dec,5))+'_var_e'
            for r in range(1,4):
                if vars()['ev'+str(r)][i] > 0:
                    namt += str(r)+', '
                    nam += str(r)
            namt += ' with eR, stl, lcn: \n'
            for r in range(1,4):
                namt += str(r)+', '+str(vars()['stl'+str(r)][i])+', '+str(vars()['lcn'+str(r)][i])+', '
            namt += str(ebmm)
            print('Savefig name: ', nam)
            ax[0].set_ylabel('Count rate (cts/s)')
            ax[int(nsr/2)].set_xlabel('\n\nTime (MJD) - 58800')
            ax[int(nsr/2)].axes.xaxis.set_ticks([])
            plt.savefig(nam+'_rebin.png', format='png')
            plt.savefig(nam+'_rebin.svg', format='svg', dpi=3200)
            plt.close()
