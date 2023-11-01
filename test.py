import smuthi.postprocessing.graphical_output as go
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
import  smuthi.postprocessing.scattered_field as sc_f



wavelength = 600
radius=129
Au_thickness=300


index_Si = opt.read_refractive_index_from_yaml('Si_Green-2008.yml', wavelength, units="nm")
index_Au = opt.read_refractive_index_from_yaml('Au_refractive_index.yml', wavelength, units="nm")

spacer = 10  # nm

layers = smuthi.layers.LayerSystem(thicknesses=[0,Au_thickness,0],
                                       refractive_indices=[1, index_Au[1] ,1])
sphere = smuthi.particles.Sphere(position=[0, 0, - radius - spacer],
                                 refractive_index=index_Si[1],
                                 radius=radius,
                                 l_max=3)

# list of all scattering particles (only one in this case)
spheres_list = [sphere]

# Initial field
plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=wavelength,
                                            polar_angle=( 25*np.pi/180),  # 25 grad to the surface
                                            azimuthal_angle=0,
                                            polarization=1,
                                            amplitude=10)  # 0=TE 1=TM


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


# go.show_near_field(quantities_to_plot=['norm(E_scat)'],
#                    show_plots=True,
#                    show_opts=[{'label': 'raw_data'},
#                               {'interpolation': 'quadric'}],
#                    save_plots=True,
#                    save_opts=[{'format': 'png'}],
#                    outputdir='./output',
#                    xmin=-1500,
#                    xmax=1500,
#                    ymax=1500,
#                    ymin=-1500,
#                    resolution_step=50,
#                    simulation=simulation,
#                    show_internal_field=True)


smuthi.postprocessing.graphical_output.show_scattering_cross_section(simulation=simulation,
                                                                     show_plots=True,
                                                                     show_opts=[{'label': 'scattering_cross_section'}],
                                                                     save_plots=False,
                                                                     save_opts=None,
                                                                     save_data=False,
                                                                     data_format='hdf5',
                                                                     outputdir='.',
                                                                     flip_downward=True,
                                                                     split=True,
                                                                     log_scale=False,
                                                                     polar_angles='default',
                                                                     azimuthal_angles='default')

