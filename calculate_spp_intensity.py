import matplotlib.pyplot as plt
import numpy as np
import smuthi
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.far_field as ff
import smuthi.utility.optical_constants as opt
import scipy.integrate as integrate
import smuthi.postprocessing.graphical_output as go


# for single wavelength
def get_extinctions(wavelength, sphere_refractive_index,
                    layers_thicknesses, layers_refractive_indeces, polarization, radius, amplitude):
    spacer = 10  # nm

    layers = smuthi.layers.LayerSystem(thicknesses=layers_thicknesses,
                                       refractive_indices=layers_refractive_indeces)

    # Scattering particle
    sphere = smuthi.particles.Sphere(position=[0, 0, radius + spacer + Au_thickness],
                                     refractive_index=sphere_refractive_index,
                                     radius=radius,
                                     l_max=3)

    # list of all scattering particles (only one in this case)
    spheres_list = [sphere]

    # Initial field
    plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=wavelength,
                                                polar_angle=(np.pi - 25 * np.pi / 180),  # 25 grad to the surface
                                                azimuthal_angle=0,
                                                polarization=polarization,
                                                amplitude=amplitude)  # 0=TE 1=TM

    # Initialize and run simulation
    simulation = smuthi.simulation.Simulation(layer_system=layers,
                                              particle_list=spheres_list,
                                              initial_field=plane_wave,
                                              solver_type='LU',
                                              solver_tolerance=1e-5,
                                              store_coupling_matrix=True,
                                              coupling_matrix_lookup_resolution=None,
                                              coupling_matrix_interpolator_kind='cubic')
    simulation.run()




    extinctions = [[], []]
    # extinctions[0] -- magnetic projections,
    # extinctions[1] -- electric projections
    # extinctions[2] -- total extinction
    for tau in [0, 1]:
        for m in [-1, 0, 1]:
            calculated_extinction = ff.extinction_cross_section(initial_field=plane_wave,
                                                                particle_list=spheres_list,
                                                                layer_system=layers, only_l=1, only_pol=polarization,
                                                                only_tau=tau, only_m=m)
            extinctions[tau].append(calculated_extinction)
    return extinctions


def add_extinctions_to_output(wavelengths, extinctions_te, extinctions_tm, te_amplitude, tm_amplitude, radius):
    px = np.zeros(len(wavelengths),dtype = 'complex_')
    py = np.zeros(len(wavelengths),dtype = 'complex_')
    pz = np.zeros(len(wavelengths),dtype = 'complex_')
    mx = np.zeros(len(wavelengths),dtype = 'complex_')
    my = np.zeros(len(wavelengths),dtype = 'complex_')
    mz = np.zeros(len(wavelengths),dtype = 'complex_')


    # Here we convert extinction from spherical to cartesian projection.
    for i in range(len(wavelengths)):
        # extinctions[i][j][k], where [i] responses for wavelength,
        # [j] responses for magnetic/electric component (0 -- magnetic, 1 -- electric, 2 -- total),
        # [k] responses for projection ([-1, 0, 1] in our case)

        mx[i] = np.sqrt((extinctions_te[i][0][0] + extinctions_te[i][0][2]).real/np.pi/radius**2) * te_amplitude

        if np.isnan(mx[i]):
            mx[i]=0

        py[i] = np.sqrt((extinctions_te[i][1][0] + extinctions_te[i][1][2]).real/np.pi/radius**2) * te_amplitude

        if np.isnan(py[i]):
            py[i]=0

        mz[i] = np.sqrt((extinctions_te[i][0][1]).real/np.pi/radius**2) * te_amplitude

        if np.isnan(mz[i]):
            mz[i]=0

        px[i] = np.sqrt((extinctions_tm[i][1][0] + extinctions_tm[i][1][2]).real/np.pi/radius**2) * np.abs(tm_amplitude)

        if np.isnan(px[i]):
            px[i]=0

        my[i] = np.sqrt((extinctions_tm[i][0][0] + extinctions_tm[i][0][2]).real/np.pi/radius**2) * np.abs(tm_amplitude)

        if np.isnan(my[i]):
            my[i]=0

        pz[i] = np.sqrt((extinctions_tm[i][1][1]).real/np.pi/radius**2) * np.abs(tm_amplitude)

        if np.isnan(pz[i]):
            pz[i]=0

    return px, py, pz, mx, my, mz


def spp_intensity(px,py,pz,mx,my,angle, index_media):

    eps=index_media**2
    kappa=-1j*np.sqrt(1/(eps+1))
    k_spp=np.sqrt(eps/(eps+1))
    intens = np.abs((mx+1j*kappa*py)*np.sin(angle) + (my - 1j * kappa * px) * np.cos(angle)-k_spp*pz)**2
    return intens


# wavelength_min = 500
# wavelength_max = 1100
# total_points = 300
# wavelengths = np.linspace(wavelength_min, wavelength_max, total_points)

wavelengths=[640, 810]

print(wavelengths[0])

index_Si = opt.read_refractive_index_from_yaml('Si_Green-2008.yml', wavelengths, units="nm")
index_Au = opt.read_refractive_index_from_yaml('Au_refractive_index.yml', wavelengths, units="nm")

index_media = 1.0
radius = 129
Au_thickness = 300

extinctions_te_gold = []
extinctions_tm_gold = []

total_amplitude=1
ellipticity=0.25
PhaseAngle=-np.pi/2
te_amplitude=ellipticity*total_amplitude

tm_amplitude=np.sqrt(1-ellipticity**2)*total_amplitude*np.exp(1j*PhaseAngle)


for i in range(len(wavelengths)):
    # With gold, TE polarization
    decomposed_te_gold_extinction = get_extinctions(wavelengths[i], index_Si[i][1],
                                                    [0, Au_thickness, 0], [index_media, index_Au[i][1], index_media], 0,
                                                    radius, te_amplitude)

    # With gold, TM polarization
    decomposed_tm_gold_extinction = get_extinctions(wavelengths[i], index_Si[i][1],
                                                    [0, Au_thickness, 0], [index_media, index_Au[i][1], index_media], 1,
                                                    radius, tm_amplitude)

    extinctions_te_gold.append(decomposed_te_gold_extinction)
    extinctions_tm_gold.append(decomposed_tm_gold_extinction)


# Now we use tau=0 for TE-polarization and tau=1 for TM,

dipoles = add_extinctions_to_output(wavelengths, extinctions_te_gold, extinctions_tm_gold, te_amplitude, tm_amplitude, radius)

numb=0
angles = np.linspace(0,2*np.pi, 300)
intensity=[]

I_spp_tot = integrate.quad(lambda angle: spp_intensity(dipoles[0][numb], dipoles[1][numb], dipoles[2][numb], dipoles[3][numb], dipoles[4][numb], angle, index_Au[numb][1]), 0, 2*np.pi)[0]

for i in angles:
    intensity.append(2*np.pi*spp_intensity(dipoles[0][numb], dipoles[1][numb], dipoles[2][numb], dipoles[3][numb], dipoles[4][numb], i, index_Au[numb][1])/I_spp_tot)


fig, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})
ax_polar.plot(angles, intensity)
#ax_polar.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax_polar.grid(True)
ax_polar.set_rmax(3)
plt.show()



