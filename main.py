import numpy as np
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.graphical_output as go
import smuthi.postprocessing.far_field as ff
import matplotlib.pyplot as plt
import smuthi.utility.optical_constants as opt
import pandas as pd
import csv


space=10
polar=1 #TM-1, TE-0
Au_thikness=300

particle_list=[96,105,129]

for ta in range(len(particle_list)):
    colour = ['red', 'green', 'blue']
    radius=particle_list[ta]
    wavelengths=np.linspace(600,1100,150)
    index_Au = opt.read_refractive_index_from_yaml('Au_refractive_index.yml', wavelengths, units="nm")
    #index_Si = opt.read_refractive_index_from_yaml('Si_Green-2008.yml', wavelengths, units="nm")


    Scat=[]
    for i in range(len(wavelengths)):
        sphere1 = smuthi.particles.Sphere(position=[0, 0, radius + space+Au_thikness],
                                          refractive_index=4,  # si sphere
                                          radius=radius,
                                          l_max=3)
        layers = smuthi.layers.LayerSystem(thicknesses=[0, Au_thikness, 0],
                                           refractive_indices=[1, index_Au[i][1], 1])
        plane_wave=smuthi.initial_field.PlaneWave(vacuum_wavelength=wavelengths[i],
                                                polar_angle= (180-25)*np.pi/180, # from top
                                                azimuthal_angle=0,
                                                polarization=polar)
        simulation = smuthi.simulation.Simulation(layer_system=layers,
                                              particle_list=[sphere1],
                                              initial_field=plane_wave,
                                              length_unit='nm',
                                              solver_type='LU',
                                              solver_tolerance=1e-5,
                                              store_coupling_matrix=True,
                                              coupling_matrix_lookup_resolution=None,
                                              coupling_matrix_interpolator_kind='cubic'
                                              )
        simulation.run()
        scs = ff.extinction_cross_section(initial_field=plane_wave,
                                              particle_list=[sphere1],
                                              layer_system=layers, only_pol=1)

        Scat.append(scs*(1e-9)**2/(np.pi*(radius*1e-9)**2))
    with open('sigma_smuti'+str(radius)+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(wavelengths, Scat))
