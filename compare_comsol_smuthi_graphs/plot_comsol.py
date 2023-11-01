import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
matplotlib.rcParams.update({'font.size': 30})
plt.rcParams.update({"font.family":"sans-serif", "font.serif":"Computer Modern", "text.usetex":True})
# plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = [12.00, 9.00]
plt.rcParams["figure.autolayout"] = True

for i in [1,2,3]:
    colour=['red','green','blue']
    radius_nm=[96,105,129]
    #multipoles = pd.read_csv("PlateData/multi"+str(radius_nm[i-1])+"-au.csv", sep=',', skiprows=4)
    sigma = pd.read_csv("comsol_simga_ext_"+str(radius_nm[i-1])+"nm_half.csv", sep=',', skiprows=4)
    lambs = np.asarray(sigma['% lambda0 (nm)'])
    sigma_ext = np.asarray(sigma['Extinction cross section (m^2)'])
    # C_ED = np.asarray(multipoles['C_ED (m^2)'])
    # C_MD = np.asarray(multipoles['C_MD (m^2)'])
    # C_EQ = np.asarray(multipoles['C_EQ (m^2)'])
    # C_MQ = np.asarray(multipoles['C_MQ (m^2)'])
    plt.plot(lambs, 2*sigma_ext/(np.pi*(radius_nm[i-1]*10**(-9))**2), c=colour[i-1],linewidth=3, label=r'\textrm{$R_'+str(i)+'='+str(radius_nm[i-1])+'$ nm}')

    # plt.plot(lambs, C_ED/(np.pi*(radius_nm*10**(-9))**2), linewidth=3, color='navy', label=r'\textrm{$C_{ED}$}')
    # plt.plot(lambs, C_MD/(np.pi*(radius_nm*10**(-9))**2), linewidth=3, color='green', label=r'\textrm{$C_{MD}$}')
    # plt.plot(lambs, C_EQ/(np.pi*(radius_nm*10**(-9))**2), linewidth=3, color='orange', label=r'\textrm{$C_{EQ}$}')
    # plt.plot(lambs, C_MQ/(np.pi*(radius_nm*10**(-9))**2), linewidth=3, color='darkviolet', label=r'\textrm{$C_{MQ}$}')

    #plt.ylim([0,max(sigma_ext)/(np.pi*(radius_nm*10**(-9))**2)])

#plt.savefig("Sigma_"+str(radius_nm)+"nm_Au_Plate.pdf", format="pdf", bbox_inches="tight")
plt.vlines(x=800, ymin=0, ymax=24 ,colors='red', ls=':', lw=4,)
plt.legend()
plt.grid()
plt.xlim([590, 1100])
plt.ylim([0,24])
plt.xlabel(r'\textrm{$\lambda$, nm}')
plt.ylabel(r'\textrm{$\sigma_{ext}$}')
plt.savefig("Sigma_for_three_Au_plate.png", format="png", bbox_inches="tight")
plt.show()