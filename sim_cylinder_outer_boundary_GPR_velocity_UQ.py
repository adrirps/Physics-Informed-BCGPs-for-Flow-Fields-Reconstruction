
# -------------------------------------------------------------------------------------------------------------
# Description
# -------------------------------------------------------------------------------------------------------------

# Physics-Informed BCGPs for Flow Fields Reconstruction : Flow around cylinder profile

# Authored by Adrian Padilla-Segarra (ONERA and INSA Toulouse) - Sep. 2025


# -------------------------------------------------------------------------------------------------------------
# Libraries
# -------------------------------------------------------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt

import core.model_tools as model_tools
import core.kernels as knp
import core.GPR as gp
import core.data_treatment as data_tools


# -------------------------------------------------------------------------------------------------------------
# Global parameters
# -------------------------------------------------------------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(description='parameter setting')

    # compact set (obstacle)
    parser.add_argument("--obstacle_type", type = str, default = 'cylinder')
    parser.add_argument("--obstacle_parameters", type = str, nargs='+', default = [0.25, 0.10, 0.025]) # [x0,y0,r] m

    # kernel
    parser.add_argument("--kernel_function", type = str, default='RBF_anisotropic')
    parser.add_argument("--kernel_parameter", type = float, nargs='+', default = [0.04,0.045,0.031])  # covariance and correlation lengths

    parser.add_argument("--base_kernel_without_BC", action ='store_true')
    parser.add_argument("--KL_measure", type = str, default = 'pushforward')
    parser.add_argument("--spectral_precision", type = str, default = 'last') # or 'grid'

    # estimation
    parser.add_argument("--plot_estimate", type = str, default = 'velocity') # or 'total_SD'

    # visualization
    parser.add_argument("--visualize", action ='store_true') # field estimations

    return  parser.parse_args()

config = get_parser()

# boundary
config.boundary_definition = {
    'top'           : 'no-slip',
    'bottom'        : 'no-slip',
    'inlet'         : 'all_data_points',
    'outlet'        : 'all_data_points',
}

# -------------------------------------------------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------------------------------------------------

model = model_tools.main_tools(config)

# -------------------------------------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------------------------------------

domain_input = np.array([[0.07,0.60],[0,0.2]])

model.internal_data = data_tools.FOAM_for_GPR(case_folder_path = r"data/cylinder_flow_Re_3000",
                                    domain_input = domain_input,
                                    time_input = [1200, 2000],
                                    reference_point = [-0.25,-0.10],
                                    config = config)
print('[Data] Internal data is loaded for computations')


# -------------------------------------------------------------------------------------------------------------
# Setting : Domain, boundary and obstacle (interpolation)
# -------------------------------------------------------------------------------------------------------------

model.define_domain(domain_input = domain_input)

t_fixed = 80 # time iteration

model.set_boundary(fixed_time = t_fixed, N_boundary_section = 30)

model.set_obstacle_points(N_obstacle_points = 1000) # for interpolation

# -------------------------------------------------------------------------------------------------------------
# Setting : Interpolation
# -------------------------------------------------------------------------------------------------------------

model.set_domain_interpolation_points(N_domain_interpolation = 'all',
                                    method = 'truth_at_data_points',
                                    fixed_time = t_fixed)

model.compute_Reynolds_number(viscosity = 1.5e-05)


# -------------------------------------------------------------------------------------------------------------
# Solution visualization
# -------------------------------------------------------------------------------------------------------------

model.plot_solution_snapshot()

# -------------------------------------------------------------------------------------------------------------
# Covariance kernel
# -------------------------------------------------------------------------------------------------------------

model.print_title('Base Kernel Initialization')

kernel = knp.Kernel(config.kernel_function,
                    parameters = config.kernel_parameter,
                    base_kernel_without_BC = config.base_kernel_without_BC)

# derivatives
kernel.load_derivatives()

# -------------------------------------------------------------------------------------------------------------
# BCGP : Boundary-constrained GP
# -------------------------------------------------------------------------------------------------------------

if hasattr(config, 'obstacle_type') and (not config.base_kernel_without_BC) :
    model.print_title('BCGP procedure')

    # Spectral method test
    X_test_decomposition = kernel.create_testing_grid(model.domain,
                                    config.obstacle_type,
                                    config.obstacle_parameters,
                                    N_test_obstacle = 20,
                                    N_test_domain = 200)

    N_integration = 2000

    kernel.obstacle_spectral_decomposition( config.obstacle_type,
                                obstacle_parameters = config.obstacle_parameters,
                                kernel_obstacle_domain = [0, 2*np.pi],
                                N_integration = N_integration,
                                KL_measure = config.KL_measure,
                                mode_precision = 1e-16,
                                testing_grid = X_test_decomposition,
                                log_to_file = False,
                                visualize = config.visualize )

# -------------------------------------------------------------------------------------------------------------
# GPR Configuration
# -------------------------------------------------------------------------------------------------------------

GP = gp.GPR(kernel, use_nugget = 'always', nugget_type = 'standard', nugget = 1e-6 ) 

# -------------------------------------------------------------------------------------------------------------
# Observation design
# -------------------------------------------------------------------------------------------------------------

# Outer boundary as observation
X_obs = model.X_boundary
velocity_obs = model.velocity_boundary

# filtering
X_obs, cond_indexes = model.filter_close_points(X_obs, tol_dist = 0.005 )
velocity_obs = velocity_obs[cond_indexes,:]

model.X_obs = X_obs
model.velocity_obs = velocity_obs

# set observations
observations_dict = {
    'structure' : 'velocity',
    'points'    : model.X_obs,
    'values'    : model.velocity_obs.flatten()
}
model.Gram_size = model.X_obs.shape[0]*2

# -------------------------------------------------------------------------------------------------------------
# GPR-based Reconstruction and Relative error computation (with visualization)
# -------------------------------------------------------------------------------------------------------------

model.perform_GPR(GP, observations_dict, out_list = 'UQ', plot_list = config.plot_estimate)


plt.show()