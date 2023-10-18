import matplotlib.pyplot as plt
import smuthi.utility.optical_constants as opt
import os
import matplotlib
import pandas as pd
import numpy as np

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
matplotlib.rcParams.update({'font.size': 30})
plt.rcParams.update({"font.family":"sans-serif", "font.serif":"Computer Modern", "text.usetex":True})
# plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = [12.00, 9.00]
plt.rcParams["figure.autolayout"] = True

particle_list=[96,105,129]

for ta in range(len(particle_list)):
    radius = particle_list[ta]
    sigma_smuthi = pd.read_csv('sigma_smuti'+str(radius)+'.csv', names=['C1','C2'])
    colour = ['red', 'green', 'blue']
    smuthi_wavelenghts = sigma_smuthi['C1']
    smuthi_sigma = sigma_smuthi['C2']
    plt.plot(smuthi_wavelenghts, smuthi_sigma, linewidth=3.0, c=colour[ta],
         label=r'\textrm{$R_' + str(ta) + '=' + str(radius) + '$ nm (SMUTHI)}', linestyle='dashed')

    #sigma = pd.read_csv("comsol_data/sigmaext" + str(radius) + "-au.csv", sep=',', skiprows=4)
    sigma = pd.read_csv("comsol_simga_ext_" + str(radius) + "nm_half.csv", sep=',', skiprows=4)
    lambs = np.asarray(sigma['% lambda0 (nm)'])
    sigma_ext = np.asarray(sigma['Extinction cross section (m^2)'])

    plt.plot(lambs, 2*sigma_ext / (np.pi * (radius * 10 ** (-9)) ** 2), c=colour[ta], linewidth=3,
             label=r'\textrm{$R_' + str(ta) + '=' + str(radius) + '$ nm (Comsol)}')

    plt.legend()
    plt.grid()
    plt.xlim([590, 1100])
    plt.ylim([0, 24])
    plt.xlabel(r'\textrm{$\lambda$, nm}')
    plt.ylabel(r'\textrm{$\sigma_{ext}$}')
    plt.savefig("Sigma_compare_smuthi_comsol-"+str(radius)+".png", format="png")
    plt.show()
#plt.show()