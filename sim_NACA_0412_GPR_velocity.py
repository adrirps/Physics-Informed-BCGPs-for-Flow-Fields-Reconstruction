
# -------------------------------------------------------------------------------------------------------------
# Description
# -------------------------------------------------------------------------------------------------------------

# Physics-Informed BCGPs for Flow Fields Reconstruction : Flow around NACA 0412 airfoil

# Authored by Adrian Padilla-Segarra (ONERA and INSA Toulouse) - Jun. 2026


# -------------------------------------------------------------------------------------------------------------
# Libraries
# -------------------------------------------------------------------------------------------------------------

import argparse, sys, time
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
    parser.add_argument("--obstacle_type", type = str, default = 'NACA_airfoil')
    parser.add_argument("--obstacle_parameters", type = str, nargs='+', default = [0,0,'0412',1]) # [x0,y0,NACA_code,chord_length]

    # kernel
    parser.add_argument("--kernel_function", type = str, default='RBF_anisotropic_additive')
    parser.add_argument("--kernel_parameter", type = float, nargs='+', default = [ 0.5, 0.1, 8, 0])  # sigma, lcor, alpha_0, M

    parser.add_argument("--without_div_free", action ='store_true')
    parser.add_argument("--base_kernel_without_BC", action ='store_true')
    parser.add_argument("--KL_measure", type = str, default = 'surface') # or 'pushforward'
    parser.add_argument("--spectral_precision", type = str, default = 'last') # or 'grid'

    # visualization
    parser.add_argument("--visualize", action ='store_true') # field estimations
    parser.add_argument("--hide_colorbar", action ='store_true')

    return  parser.parse_args()

config = get_parser()

# boundary
config.boundary_definition = {
    'top'           : 'all_data_points',
    'bottom'        : 'all_data_points',
    'inlet'         : 'all_data_points',
    'outlet'        : 'all_data_points',
}

# -------------------------------------------------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------------------------------------------------

model = GPR_model(config)
model.v_map = 'turbo' # for velocity plots

# -------------------------------------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------------------------------------

domain_input = np.array([[-0.08,0.15],[-0.1,0.1]]) # leading edge

model.internal_data = data_tools.FOAM_for_GPR(case_folder_path = r"data\NACA_0412_Re_3000_flow",
                                    domain_input = domain_input,
                                    time_input = [1200, 2250],
                                    reference_point = [0,0],
                                    boundary_source_precision = 0.002,
                                    config = config)
print('[Data] Internal data is loaded for computations')


# -------------------------------------------------------------------------------------------------------------
# Setting : Domain, boundary and obstacle (interpolation)
# -------------------------------------------------------------------------------------------------------------


model.define_domain(domain_input = domain_input)

t_fixed = 3 # time iteration

model.set_boundary(fixed_time = t_fixed)

model.set_obstacle_points(N_obstacle_points = 5000) # for interpolation

# -------------------------------------------------------------------------------------------------------------
# Setting : Interpolation
# -------------------------------------------------------------------------------------------------------------


model.set_domain_interpolation_points(N_domain_interpolation = 2000,
                                    method = 'truth_at_data_points',
                                    fixed_time = t_fixed,
                                    airfoil_box_distance_interpolation = 0.012)

model.compute_Reynolds_number(viscosity = 1.55e-05)

# -------------------------------------------------------------------------------------------------------------
# Solution visualization
# -------------------------------------------------------------------------------------------------------------

model.plot_u_max = 0.075
model.plot_solution_snapshot()

u_truth_profile = model.get_true_fields(0, model.X_obstacle, out_list = 'velocity')['velocity']

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

    X_test_decomposition = kernel.create_testing_grid(model.domain,
                                    config.obstacle_type,
                                    config.obstacle_parameters,
                                    N_test_obstacle = 20,
                                    N_test_domain = 200)

    N_integration = 300

    kernel.obstacle_spectral_decomposition( config.obstacle_type,
                                obstacle_parameters = config.obstacle_parameters,
                                kernel_obstacle_domain = [2.3, 4], # NACA leading edge
                                N_integration = N_integration,
                                KL_measure = config.KL_measure,
                                mode_precision = 1e-16,
                                testing_grid = X_test_decomposition,
                                log_to_file = False,
                                visualize = config.visualize )

# -------------------------------------------------------------------------------------------------------------
# GPR Configuration
# -------------------------------------------------------------------------------------------------------------

if config.kernel_parameter[3] == 0 :
    nugget_val = 1e-8
else :
    nugget_val = 1e-10

GP = gp.GPR(kernel, use_nugget = 'always', nugget_type = 'standard', nugget = nugget_val )

# -------------------------------------------------------------------------------------------------------------
# Observation design
# -------------------------------------------------------------------------------------------------------------

# set observation grid
X_obs = model.build_noised_grid_domain(seed = 1287, N_grid = 300)

# filter points close to airfoil
X_obs, idx_temp = model.filter_airfoil_box(X_obs, distance_tol = 0.007)

# get training values
velocity_obs = model.get_true_fields(0, X_obs, out_list = 'velocity')['velocity']

# save for plot
model.X_obs = X_obs
print(f'[Design] Observations set at computational domain: {X_obs.shape[0]}')

# set observations
observations_dict = {
    'structure' : 'velocity',
    'points'    : X_obs,
    'values'    : velocity_obs.flatten()
}
model.Gram_size = X_obs.shape[0]*2

# -------------------------------------------------------------------------------------------------------------
# GPR-based Reconstruction (with visualization)
# -------------------------------------------------------------------------------------------------------------

rel_agg_obstacle_normal = []
rel_agg_obstacle_tangent = []
abs_scalar_stream = []
spectral_precision = []

# set spectral precision

if config.spectral_precision == 'last' :
    spectral_precision_integers = np.array([12])
elif config.spectral_precision == 'grid' :
    spectral_precision_integers = np.arange(9, 15)
tol_grid = 1/(10**spectral_precision_integers.astype(float))

for it_tol in tol_grid :

    # set kernel truncation

    if not config.base_kernel_without_BC :
        spectral_precision_n = kernel.set_mode_truncation(precision = it_tol,
                                                        KL_measure = config.KL_measure,
                                                        N_integration = N_integration)
        spectral_precision_n = - np.log10(spectral_precision_n)
        spectral_precision.append(spectral_precision_n)

    # Perform estimations

    start_time_mean = time.time()

    if not config.without_div_free :
        out_dict = model.perform_GPR(GP, observations_dict, out_list = ['domain', 'obstacle' ,'stream_obstacle'], plot_list = ['velocity'])

        abs_scalar_stream.append(np.abs(out_dict['stream_obstacle']))

    else :
        out_dict = model.vector_valued_GPR(GP, observations_dict, out_list = ['velocity', 'obstacle'], plot_list = ['velocity'])

    end_time_mean = time.time()

    print(f'[Time] Velocity posterior mean CPU time: {end_time_mean - start_time_mean:.3f} seconds')


    # divergence fit

    div_val = model.compute_divergence(GP, observations_dict, eps = 1e-6)

    plot_scale = 3.1
    model.do_plot({'u_error' : div_val }, method = 'u_error', plot_min = - plot_scale, plot_max = plot_scale,
                  save_name = 'velocity_divergence', cmap = 'seismic', title = False )

    eps_div, _ = model.compute_space_integral(model.X_domain, np.abs(div_val), dom_relative = True )


    # compute profile boundary indicator (relative aggregated error on compact set)

    rel_agg_obstacle_normal_n, rel_agg_obstacle_tangent_n = model.compute_obstacle_normal_error(out_dict['obstacle'], relative = True)
    rel_agg_obstacle_normal.append(rel_agg_obstacle_normal_n)
    rel_agg_obstacle_tangent.append(rel_agg_obstacle_tangent_n)

    print(f'[Model] Relative-agg. indicator of velocity normal components on obstacle : {rel_agg_obstacle_normal}')

if config.spectral_precision == 'grid' :

    # -------------------------------------------------------------------------------------------------------------
    # Convergence of profile indicators
    # -------------------------------------------------------------------------------------------------------------

    if not config.base_kernel_without_BC :
        model.plot_profile_indicators(spectral_precision, rel_agg_obstacle_normal, abs_scalar_stream )

    plt.show()
    sys.exit()

# -------------------------------------------------------------------------------------------------------------
# Compute and plot residuals
# -------------------------------------------------------------------------------------------------------------

u_residual = out_dict['u_domain'] - model.velocity_truth

# RMSE of test points
RMSE = np.sqrt( (u_residual**2).mean() )

# plots
u_residual_norm = np.linalg.norm( u_residual , axis = 1)
print(f'[Visualization] Residual max plot: {u_residual_norm.max()}')
model.do_plot({'u_error' : u_residual_norm }, method = 'u_error', plot_min = 0.0, plot_max = 0.035, save_name = 'velocity_residual')

# -------------------------------------------------------------------------------------------------------------
# Uncertainty Quantification
# -------------------------------------------------------------------------------------------------------------

model.print_title('UQ computation')

start_time_cov = time.time()

if not config.without_div_free :
    u_var = GP.interpolation('velocity_covariance', model.X_domain, observations_dict)
else :
    out_var = model.vector_valued_GPR(GP, observations_dict, out_list = ['velocity_covariance'])
    u_var = out_var['u_covariance']

end_time_cov = time.time()

print(f'[Time] Velocity posterior covariance CPU time: {end_time_cov - start_time_cov:.3f} seconds')


var_trace = np.diag(u_var)[::2] + np.diag(u_var)[1::2]
total_SD = np.sqrt( var_trace )

# plot
model.do_plot({ 'UQ_field' : np.log10(total_SD) }, method = 'UQ_field', plot_min = -7.0, plot_max = -1.0, save_name = 'velocity_total_SD_log')

# UQ obstacle verification
u_truth_profile = model.get_true_fields(0, model.X_obstacle, out_list = 'velocity')['velocity']

if not config.without_div_free :
    u_mean_obs = GP.interpolation('velocity_mean', model.X_obstacle, observations_dict).reshape((model.X_obstacle.shape[0],2))
    u_var_check = GP.interpolation('velocity_covariance', model.X_obstacle, observations_dict)
else :
    out_var_obs = model.vector_valued_GPR(GP, observations_dict, out_list = ['obstacle','velocity_covariance'], X_new = model.X_obstacle)
    u_mean_obs = out_var_obs['obstacle']
    u_var_check = out_var_obs['u_covariance']

u_var_trace_check = np.diag(u_var_check)[::2] + np.diag(u_var_check)[1::2]
u_total_SD_check = np.sqrt(u_var_trace_check)

# plots
for i in range(2) :
    fig = plt.figure()
    plt.plot(model.gamma_grid, u_truth_profile[:,i], 'k--', label = 'truth')
    plt.plot(model.gamma_grid, u_mean_obs[:,i], label = 'posterior mean')
    plt.fill_between(model.gamma_grid,
                     u_mean_obs[:,i] - 1.96*np.sqrt(np.diag(u_var_check)[i::2]),
                     u_mean_obs[:,i] + 1.96*np.sqrt(np.diag(u_var_check)[i::2]), color='gray', alpha=0.3, label="95% confidence interval")
    if i == 0 :
        plt.ylim([-0.005, 0.055])
        plt.legend(loc="lower right", fontsize = 'small')
    elif i == 1 :
        plt.ylim([-0.03, 0.03])
        plt.legend(loc="upper right", fontsize = 'small')

    plt.xlim([2.35, 3.93])

    plt.savefig(r".\BCGP_figures\\" + 'UQ_component_' + str(int(i+1)) + '.pdf', bbox_inches='tight', pad_inches=0.01)

    # -------------------------------------------------------------------------------------------------------------
    # Prediction interval coverage
    # -------------------------------------------------------------------------------------------------------------

    u_coverage_i = np.abs(u_residual[:,i]) / (1.96*np.sqrt(np.diag(u_var)[i::2]))
    fraction_i = np.sum(u_coverage_i <= 1.0)/ u_coverage_i.shape[0]
    print(f'[Results] Velocity component {i+1} coverage: {fraction_i}')

print(f'[Results] Relative-agg. indicator of velocity normal components on obstacle  : {rel_agg_obstacle_normal[0]}')
print(f'[Results] RMSE of {u_residual.shape[0]} test points: {RMSE}')
print(f'[Results] L1-norm of velocity divergence: {eps_div}')


plt.show()
