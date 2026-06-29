
# -------------------------------------------------------------------------------------------------------------
# Description
# -------------------------------------------------------------------------------------------------------------

# Physics-Informed BCGPs for Flow Fields Reconstruction : Flow around cylinder profile

# Authored by Adrian Padilla-Segarra (ONERA and INSA Toulouse) - Jun. 2026


# -------------------------------------------------------------------------------------------------------------
# Libraries
# -------------------------------------------------------------------------------------------------------------

import argparse, os, dill
import numpy as np
import matplotlib.pyplot as plt

from core.models import GPR_model
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

    parser.add_argument("--base_kernel_without_BC", action ='store_true')
    parser.add_argument("--KL_measure", type = str, default = 'pushforward')
    parser.add_argument("--spectral_precision", type = str, default = 'last') # or 'grid'

    # visualization
    parser.add_argument("--visualize", action ='store_true') # field estimations
    parser.add_argument("--hide_colorbar", action ='store_true')

    return  parser.parse_args()

save_folder = r".\BCGP_figures"
os.makedirs(save_folder, exist_ok = True)

config = get_parser()

# boundary
config.boundary_definition = {
    'top'           : 'no-slip',
    'bottom'        : 'no-slip',
    'inlet'         : 'all_data_points',
    'outlet'        : 'all_data_points',
}

sigma_grid =  np.array([0.01, 0.05, 0.5])
lcor_grid = np.linspace(0.04, 0.2, 50) # vertical scale

# -------------------------------------------------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------------------------------------------------

model = GPR_model(config)

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

t_fixed = 80 # simulation iteration

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
# Observation design
# -------------------------------------------------------------------------------------------------------------

if config.base_kernel_without_BC :
    N_domain_observation_points = 300
    N_normal_observation_discrete = 142
    N_local_region = 0

else : # with BCGP
    N_domain_observation_points = 500
    N_normal_observation_discrete = 0
    N_local_region = 30
    local_region_domain = np.array([[0.28,0.38],[0.08,0.12]])

# set observation grid
X_domain_sub_raw = model.build_grid(N_domain_observation_points, model.domain, with_limits = False)
np.random.seed(1212)
X_domain_sub = X_domain_sub_raw + np.random.normal(0, 0.005, X_domain_sub_raw.shape[0]*2).reshape(X_domain_sub_raw.shape) # noise

# get velocity values
X_domain_sub, velocity_domain_sub = model.internal_data.get_domain_points(X_grid_particular = X_domain_sub)
X_domain_sub['all'], indexes_domain = model.points_inside_domain(X_domain_sub['all'])
velocity_domain_sub['all'] = velocity_domain_sub['all'][:, indexes_domain,:]

# set observation objects
X_obs = np.vstack((model.X_boundary, X_domain_sub['all']))
velocity_obs = np.vstack((model.velocity_boundary, velocity_domain_sub['all'][t_fixed,:,:]))

# filter close points
inter_distance = 0.01 # lower bound
X_obs, cond_indexes = model.filter_close_points(X_obs, inter_distance)
velocity_obs = velocity_obs[cond_indexes,:]

# add specific region
if N_local_region > 0 :
    X_local_region, tol_min = model.set_local_region('box', N_local_region, local_region_domain)
    velocity_region = model.get_true_fields(t_fixed, X_local_region, out_list = 'velocity')['velocity']
    X_obs = np.vstack((X_local_region, X_obs))
    velocity_obs = np.vstack((velocity_region, velocity_obs))

# filter close points
try: tol_min
except: tol_min = 1e-10
X_obs, cond_indexes_local = model.filter_close_points(X_obs, tol_min)
velocity_obs = velocity_obs[cond_indexes_local,:]

model.X_obs = X_obs
model.velocity_obs = velocity_obs

# set observations
observations_dict = {
    'structure' : 'velocity',
    'points'    : model.X_obs,
    'values'    : model.velocity_obs.flatten()
}
model.Gram_size = model.X_obs.shape[0]*2

if N_normal_observation_discrete > 0 :

    # add discrete boundary condition as observations (without BCGP)

    model.X_normal_obs, normal_vectors = model.add_discrete_obstacle_observation('discrete_normal', N_normal_observation_discrete)
    observations_dict['structure'] = 'velocity_and_normal'
    observations_dict['points_normal'] = model.X_normal_obs
    observations_dict['normal_vectors'] = normal_vectors
    observations_dict['values_normal'] = np.zeros((model.X_normal_obs.shape[0],))
    model.Gram_size += model.X_normal_obs.shape[0]

# -------------------------------------------------------------------------------------------------------------
# GPR-based Reconstruction and Relative error computation (with visualization)
# -------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------
#   Sensibility grid
# -------------------------------------------------------------------------------------------------------------

N_sigma = sigma_grid.shape[0]
N_lcor = lcor_grid.shape[0]
sigma_mesh, lcor_mesh = np.meshgrid(sigma_grid, lcor_grid)
sigma_mesh = sigma_mesh.flatten()
lcor_mesh = lcor_mesh.flatten()
N_gs = sigma_mesh.shape[0]

print(f'Grid size: {N_gs} nodes')

rel_agg_obstacle_normal = np.zeros((N_gs,))

for it_gs in range(N_gs) :

    model.print_title(f'Grid iteration - sigma = {sigma_mesh[it_gs]:.3f} - lcor = {lcor_mesh[it_gs]:.3f}')

    kernel_parameter_grid_n = np.zeros((3))
    kernel_parameter_grid_n[0] = sigma_mesh[it_gs]
    kernel_parameter_grid_n[1] = lcor_mesh[it_gs]
    kernel_parameter_grid_n[2] = lcor_mesh[it_gs] / 1.5  # from anisotropy


    # -------------------------------------------------------------------------------------------------------------
    # Covariance kernel
    # -------------------------------------------------------------------------------------------------------------

    model.print_title('Base Kernel Initialization')

    kernel = knp.Kernel(config.kernel_function,
                        parameters = kernel_parameter_grid_n,
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

        N_integration = 400

        kernel.obstacle_spectral_decomposition( config.obstacle_type,
                                    obstacle_parameters = config.obstacle_parameters,
                                    kernel_obstacle_domain = [0, 2*np.pi],
                                    N_integration = N_integration,
                                    KL_measure = config.KL_measure,
                                    mode_precision = 1e-12,
                                    testing_grid = X_test_decomposition,
                                    log_to_file = False,
                                    visualize = False )

    # -------------------------------------------------------------------------------------------------------------
    # GPR Configuration
    # -------------------------------------------------------------------------------------------------------------

    GP = gp.GPR(kernel, use_nugget = 'always', nugget_type = 'standard', nugget = 1e-6 )


    # -------------------------------------------------------------------------------------------------------------
    # Perform estimations
    # -------------------------------------------------------------------------------------------------------------

    out_dict = model.perform_GPR(GP, observations_dict, out_list = ['domain', 'obstacle'], plot_list = 'velocity')

    # compute relative aggregated error

    rel_agg_obstacle_normal_n, rel_agg_obstacle_tangent_n = model.compute_obstacle_normal_error(out_dict['obstacle'], relative = True)
    rel_agg_obstacle_normal[it_gs] = rel_agg_obstacle_normal_n

    print(f'[Model] Relative-agg. normal components of velocity around obstacle  : {rel_agg_obstacle_normal_n}')

rel_agg_obstacle_normal = rel_agg_obstacle_normal.reshape((N_lcor, N_sigma))

save_file_path_base = os.path.join(save_folder, 'Base_kernel_obstacle_normal.pkl')

if config.base_kernel_without_BC :

    # Save base kernel estimations
    with open(save_file_path_base, 'wb') as save_file:
        dill.dump(rel_agg_obstacle_normal, save_file)

    # Run script with BCGP
    import subprocess

    model.print_title('[Sim] Running BCGP comparison')
    subprocess.run(["py", os.path.abspath(__file__)])


else:

    # save results from BCGP kernel
    with open(os.path.join(save_folder, 'BCGP_kernel_obstacle_normal.pkl'), 'wb') as save_file:
        dill.dump(rel_agg_obstacle_normal, save_file)

    # load results from base kernel 
    with open(save_file_path_base, 'rb') as load_file:
        rel_agg_obstacle_normal_base = dill.load(load_file)

    # Sensibility comparison
    for it_sig in range(sigma_grid.shape[0]) :

        plt.figure(figsize=(5, 4))
        plt.semilogy(lcor_grid, rel_agg_obstacle_normal[:,it_sig], '-', c = 'black', label = 'BCGP')
        plt.semilogy(lcor_grid, rel_agg_obstacle_normal_base[:,it_sig], '--', c='black', label = 'Base GP')            
        plt.xlabel("Correlation length")
        plt.ylabel("Normal profile indicator", rotation = 90)
        plt.grid(True, which="minor", alpha = 0.6, ls = '--')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(save_folder, 'compare_lcor_sig' + str(it_sig) + '.pdf'), bbox_inches='tight', pad_inches=0.01)


    plt.show()
