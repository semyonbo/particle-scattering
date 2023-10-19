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
import smuthi.postprocessing.graphical_output as go
import io
import yaml
import os
import matplotlib

smuthi.utility.cuda.enable_gpu()


#os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
matplotlib.rcParams.update({'font.size': 30})
#plt.rcParams.update({"font.family":"sans-serif", "font.serif":"Computer Modern", "text.usetex":True})
# plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = [12.00, 9.00]
plt.rcParams["figure.autolayout"] = True


# for single wavelength
def get_extinctions(wavelength, sphere_refractive_index,
                    layers_thicknesses, layers_refractive_indeces, polarization, radius):
    spacer = 10  # nm

    layers = smuthi.layers.LayerSystem(thicknesses=layers_thicknesses,
                                       refractive_indices=layers_refractive_indeces)
    sphere = smuthi.particles.Sphere(position=[0, 0, radius + spacer + Au_thickness],
                                     refractive_index=sphere_refractive_index,
                                     radius=radius,
                                     l_max=3)

    # list of all scattering particles (only one in this case)
    spheres_list = [sphere]

    # Initial field
    plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=wavelength,
                                                polar_angle=(np.pi - 25 * np.pi / 180),
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
                                              coupling_matrix_interpolator_kind='cubic'
                                              )
    simulation.run()

    # Since this moment we calculate extinction
    # Calculate total extinction of dipole (another multipoles are cut by only_l=1)
    general_ecs = ff.extinction_cross_section(initial_field=plane_wave,
                                              particle_list=spheres_list,
                                              layer_system=layers, only_pol=polarization)

    extinctions = [[], [], []]
    quadrupoles=[[],[]]
    # extinctions[0] -- magnetic projections,
    # extinctions[1] -- electric projections
    # extinctions[2] -- total extinction
    for tau in [0, 1]:
        for m in [-1, 0, 1]:
            calculated_extinction = ff.extinction_cross_section(initial_field=plane_wave,
                                                                particle_list=spheres_list,
                                                                layer_system=layers, only_pol=polarization,
                                                                only_tau=tau, only_m=m, only_l=1)
            extinctions[tau].append(calculated_extinction)
    extinctions[2].append(general_ecs)
    for tau in [0, 1]:
        for m in [-1, 0, 1]:
            calculated_extinction_q = ff.extinction_cross_section(initial_field=plane_wave,
                                                                particle_list=spheres_list,
                                                                layer_system=layers, only_pol=polarization,
                                                                only_tau=tau, only_m=m, only_l=2)
            quadrupoles[tau].append(calculated_extinction_q)

    return extinctions, quadrupoles

def add_extinctions_to_output(wavelengths, extinctions, quadrupoles, radius):
    #def add_extinctions_to_output(wavelengths, extinctions, quadrupoles, radius, tau,
    #                              color_for_single_line, color_for_twin_lines,
    #                              x_description, y_description, z_description):
    # x_component = np.zeros(len(extinctions))
    # z_component = np.zeros(len(extinctions))
    # y_component = np.zeros(len(extinctions))
    total = np.zeros(len(extinctions))

    Q_ED = np.zeros(len(wavelengths))
    Q_MD = np.zeros(len(wavelengths))
    Q_EQ = np.zeros(len(wavelengths))
    Q_MQ = np.zeros(len(wavelengths))

    for i in range(len(wavelengths)):
        Q_ED[i] = (extinctions[i][1][0] + extinctions[i][1][2] + extinctions[i][1][1]).real / np.pi / radius ** 2
        Q_MD[i] = (extinctions[i][0][0] + extinctions[i][0][2] + extinctions[i][0][1]).real / np.pi / radius ** 2
        Q_MQ[i] = (quadrupoles[i][0][0] + quadrupoles[i][0][2] + quadrupoles[i][0][1]).real / np.pi / radius ** 2
        Q_EQ[i] = (quadrupoles[i][1][0] + quadrupoles[i][1][2] + quadrupoles[i][1][1]).real / np.pi / radius ** 2
        total[i] = (extinctions[i][2][0]).real / np.pi / radius ** 2
    # Here we convert extinction from spherical to cartesian projection.
    # for i in range(len(extinctions)):
    #     # extinctions[i][j][k], where [i] responses for wavelength,
    #     # [j] responses for magnetic/electric component (0 -- magnetic, 1 -- electric, 2 -- total),
    #     # [k] responses for projection ([-1, 0, 1] in our case)
    #     x_component[i] = (extinctions[i][tau][0] + extinctions[i][tau][2]).real / np.pi / radius ** 2
    #
    #     y_component[i] = (extinctions[i][1 - tau][0] + extinctions[i][1 - tau][2]).real / np.pi / radius ** 2
    #
    #     z_component[i] = (extinctions[i][tau][1]).real / np.pi / radius ** 2
    #
    #     total[i] = (extinctions[i][2][0]).real / np.pi / radius ** 2

    # plt.plot(wavelengths, x_component, linestyle='dashed', color=color_for_twin_lines, label=x_description)
    # plt.plot(wavelengths, y_component, linestyle='dashed', color=color_for_single_line,label=y_description)
    # plt.plot(wavelengths, z_component, color=color_for_twin_lines, label=z_description)
    # plt.plot(wavelengths, total, color='black', label='total')
    # plt.legend()
    plt.plot(wavelengths, Q_ED, linestyle='dashed', color='red', label=r'$Q_{ED}$', lw=3)
    plt.plot(wavelengths, Q_MD, linestyle='dashed', color='blue', label=r'$Q_{MD}$', lw=3)
    plt.plot(wavelengths, Q_MQ, color='green', label=r'$Q_{MQ}$', lw=3)
    plt.plot(wavelengths, Q_EQ, color='brown', label=r'$Q_{EQ}$', lw=3)
    plt.plot(wavelengths, total, color='black', label=r'$Q_{tot}$', lw=3)
    plt.legend()


wavelength_min = 500
wavelength_max = 1100
total_points = 300
wavelengths = np.linspace(wavelength_min, wavelength_max, total_points)

index_Si = opt.read_refractive_index_from_yaml('Si_Green-2008.yml', wavelengths, units="nm")
index_Au = opt.read_refractive_index_from_yaml('Au_refractive_index.yml', wavelengths, units="nm")

index_media = 1.0
radius = 129
Au_thickness = 300

extinctions_te_gold = []
extinctions_tm_gold = []
extinctions_te_gold_q = []
extinctions_tm_gold_q = []

for i in range(len(wavelengths)):
    # With gold, TM polarization
    decomposed_tm_gold_extinction = get_extinctions(wavelengths[i], 4,
                                                    [0, Au_thickness, 0], [index_media, index_Au[i][1], index_media], 1,
                                                    radius)[0]
    decomposed_tm_gold_extinction_q = get_extinctions(wavelengths[i], 4,
                                                    [0, Au_thickness, 0], [index_media, index_Au[i][1], index_media], 1,
                                                    radius)[1]


    extinctions_tm_gold.append(decomposed_tm_gold_extinction)
    extinctions_tm_gold_q.append(decomposed_tm_gold_extinction_q)

# Since this moment we just output the calculated extinction decompositions.
add_extinctions_to_output(wavelengths, extinctions_tm_gold, extinctions_tm_gold_q, radius)

plt.xlim([500, 1100])
plt.ylim([-5,25])
plt.grid()
plt.xlabel(r'$\lambda$, nm')
plt.ylabel(r'$\sigma_{ext}$')
plt.savefig("Multipoles_SMUTHI.png", format="png", bbox_inches="tight")
plt.show()


