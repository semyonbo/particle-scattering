import matplotlib.pyplot as plt
import unittest
import numpy as np
import smuthi
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.far_field as ff
import smuthi.utility.optical_constants as opt
import io
import yaml


# for single wavelength
def get_extinctions(wavelength, sphere_refractive_index,
                    layers_thicknesses, layers_refractive_indeces, polarization, radius):
    spacer = 10  # nm

    layers = smuthi.layers.LayerSystem(thicknesses=layers_thicknesses,
                                       refractive_indices=layers_refractive_indeces)

    # Scattering particle
    sphere = smuthi.particles.Sphere(position=[0, 0, -radius - spacer],
                                     refractive_index=sphere_refractive_index,
                                     radius=radius,
                                     l_max=3)

    # list of all scattering particles (only one in this case)
    spheres_list = [sphere]

    # Initial field
    plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=wavelength,
                                                polar_angle=(25 * np.pi / 180),  # 25 grad to the surface
                                                azimuthal_angle=0,
                                                polarization=polarization)  # 0=TE 1=TM

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

    # Since this moment we calculate extinction
    # Calculate total extinction

    #scat_cs=ff.total_scattering_cross_section(initial_field=plane_wave,
    #                                  particle_list=spheres_list,
    #                                  layer_system=layers)
    general_ecs = ff.extinction_cross_section(initial_field=plane_wave,
                                              particle_list=spheres_list,
                                              layer_system=layers, only_pol=polarization)

    extinctions = [[], [[],[]], [[],[]]]
    # ord - order of multipole (1 - dipole, 2 - quadrupole)
    # extinctions[ord][0] -- magnetic projections,
    # extinctions[ord][1] -- electric projections
    # extinctions[0] -- total extinction
    for order in [1,2]:
        for tau in [0, 1]:
            calculated_extinction = ff.extinction_cross_section(initial_field=plane_wave,
                                                                    particle_list=spheres_list,
                                                                    layer_system=layers, only_l=order, only_pol=polarization,
                                                                    only_tau=tau)
            extinctions[order][tau].append(calculated_extinction)
    extinctions[0].append(general_ecs)

    return extinctions


def add_extinctions_to_output(wavelengths, extinctions, subplot, radius,
                              color_for_electric, color_for_magnetic):
    Q_ED = np.zeros(len(extinctions))
    Q_MD = np.zeros(len(extinctions))
    Q_EQ = np.zeros(len(extinctions))
    Q_MQ = np.zeros(len(extinctions))
    total = np.zeros(len(extinctions))

    # Here we convert extinction from spherical to cartesian projection.
    for i in range(len(extinctions)):
        Q_ED[i]=(extinctions[i][1][0][0]).real/np.pi/radius**2
        Q_MD[i] = (extinctions[i][1][1][0]).real/np.pi/radius**2
        Q_EQ[i] = (extinctions[i][2][0][0]).real/np.pi/radius**2
        Q_MQ[i] = (extinctions[i][2][1][0]).real/np.pi/radius**2
        total[i] = (extinctions[i][0][0]).real/np.pi/radius**2

    subplot.plot(wavelengths, Q_ED, color=color_for_electric, label='$\sigma_{ED}$')
    subplot.plot(wavelengths, Q_MD, color=color_for_magnetic, label='$\sigma_{MD}$')
    subplot.plot(wavelengths, Q_EQ, linestyle='dashed', color=color_for_electric, label='$\sigma_{EQ}$')
    subplot.plot(wavelengths, Q_MQ, linestyle='dashed', color=color_for_magnetic, label='$\sigma_{MQ}$')
    subplot.plot(wavelengths, total, color='black', label='$\sigma_{tot}$')
    subplot.legend()


wavelength_min = 500
wavelength_max = 1100
total_points = 300
wavelengths = np.linspace(wavelength_min, wavelength_max, total_points)

index_Si = opt.read_refractive_index_from_yaml('Si_Green-2008.yml', wavelengths, units="nm")
index_Au = opt.read_refractive_index_from_yaml('Au_refractive_index.yml', wavelengths, units="nm")

index_media = 1.0
radius = 129
Au_thickness = 300

extinctions_tm_gold = []

for i in range(len(wavelengths)):

    # With gold, TM polarization
    decomposed_tm_gold_extinction = get_extinctions(wavelengths[i], index_Si[i][1],
                                                    [0, Au_thickness, 0], [index_media, index_Au[i][1], index_media], 1,
                                                    radius)
    extinctions_tm_gold.append(decomposed_tm_gold_extinction)

# Since this moment we just output the calculated extinction decompositions.
f, ax_1 = plt.subplots(1, 1, sharex=True)

plt.annotate(text="TM", xy=(0.5, 0.9), xycoords='figure fraction')


add_extinctions_to_output(wavelengths, extinctions_tm_gold, ax_1, radius, 'red', 'blue')
ax_1.set_ylim(ymin=-5, ymax=25)
plt.show()
f.savefig("sigma_ext_smuthi_multupoles.pdf", bbox_inches='tight')
