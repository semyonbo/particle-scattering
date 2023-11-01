import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate


def spp_intensity(px,py,pz,mx,my,angle, index_media):

    eps=index_media**2
    kappa=-1j*np.sqrt(1/(eps+1))
    k_spp=np.sqrt(eps/(eps+1))
    intens = np.abs((mx+1j*kappa*py)*np.sin(angle) + (my - 1j * kappa * px) * np.cos(angle)-k_spp*pz)**2
    return intens

px=1*np.exp(-2j)
py=1*np.exp(2j)
pz=1*np.exp(2j)
mx=1*np.exp(2j)
my=1*np.exp(2j)

index_Au=0.3+3j

angles = np.linspace(0,2*np.pi, 300)
intensity=[]

I_spp_tot = integrate.quad(lambda angle: spp_intensity(px, py,pz,mx,my, angle, index_Au), 0, 2*np.pi)[0]

for i in angles:
    intensity.append(2*np.pi*spp_intensity(px,py,pz,mx,my, i, index_Au)/I_spp_tot)


fig, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})
ax_polar.plot(angles, intensity)
#ax_polar.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax_polar.grid(True)
ax_polar.set_rmax(3)
plt.show()
