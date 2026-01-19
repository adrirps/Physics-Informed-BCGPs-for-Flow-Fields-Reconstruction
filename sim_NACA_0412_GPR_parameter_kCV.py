
# -------------------------------------------------------------------------------------------------------------
# Description
# -------------------------------------------------------------------------------------------------------------

# Physics-Informed BCGPs for Flow Fields Reconstruction : Flow around NACA 0412 airfoil

# Authored by Adrian Padilla-Segarra (ONERA and INSA Toulouse) - Jan. 2026


# -------------------------------------------------------------------------------------------------------------
# Libraries
# -------------------------------------------------------------------------------------------------------------

import argparse
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
    parser.add_argument("--additive_scales", type = int, default = 0) # parameter M

    parser.add_argument("--base_kernel_without_BC", action ='store_true')
    parser.add_argument("--KL_measure", type = str, default = 'surface') # or 'pushforward'

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

if config.additive_scales == 0 :
    nugget_val = 1e-8
else :
    nugget_val = 1e-10

sigma_grid =  np.array([0.005,0.01,0.05,0.1,0.15, 0.2])
lcor_grid = np.array([0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])


# -------------------------------------------------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------------------------------------------------

model = GPR_model(config)

# -------------------------------------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------------------------------------

domain_input = np.array([[-0.08,0.15],[-0.1,0.1]]) # leading edge

model.internal_data = data_tools.FOAM_for_GPR(case_folder_path = r"data\NACA_0412_Re_3000_flow",
                                    domain_input = domain_input,
                                    time_input = [1200, 6000],
                                    reference_point = [0,0],
                                    boundary_source_precision = 0.002,
                                    config = config)
print('[Data] Internal data is loaded for computations')


# -------------------------------------------------------------------------------------------------------------
# Setting : Domain, boundary and obstacle (interpolation)
# -------------------------------------------------------------------------------------------------------------


model.define_domain(domain_input = domain_input)

t_fixed = 3 # simulation iteration

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

model.plot_solution_snapshot()


# -------------------------------------------------------------------------------------------------------------
# Observation design
# -------------------------------------------------------------------------------------------------------------

# set observation grid
X_obs = model.build_noised_grid_domain(seed = 1287, N_grid = 300)

# filter points close to airfoil
X_obs, idx_temp = model.filter_airfoil_box(X_obs, distance_tol = 0.007)

velocity_obs = model.get_true_fields(0, X_obs, out_list = 'velocity')['velocity']

X_obs_for_CV = X_obs.copy()
values_obs_for_CV = velocity_obs.copy()

# -------------------------------------------------------------------------------------------------------------
# Hyperparameters Gridsearch with CV
# -------------------------------------------------------------------------------------------------------------

N_sigma = sigma_grid.shape[0]
N_lcor = lcor_grid.shape[0]
sigma_mesh, lcor_mesh = np.meshgrid(sigma_grid, lcor_grid)
sigma_mesh = sigma_mesh.flatten()
lcor_mesh = lcor_mesh.flatten()
N_gs = sigma_mesh.shape[0]

print(f'Gridsearch size: {N_gs} nodes')


# -------------------------------------------------------------------------------------------------------------
# k-folds for Cross Validation
# -------------------------------------------------------------------------------------------------------------

k_folds = 4

N_obs = X_obs_for_CV.shape[0]
seed_cv = 847

rng = np.random.default_rng(seed_cv)
idx_obs = rng.permutation(N_obs)
idx_folds = np.array_split(idx_obs, k_folds)

p0_loss_gs = np.zeros((N_gs, k_folds))
p1_loss_gs = np.zeros((N_gs, k_folds))

for it_gs in range(N_gs) :

    kernel_parameter_grid_search_n = np.zeros((4))
    kernel_parameter_grid_search_n[0] = sigma_mesh[it_gs]
    kernel_parameter_grid_search_n[1] = lcor_mesh[it_gs]
    kernel_parameter_grid_search_n[2] = 8 # anisotropy at coarse scale
    kernel_parameter_grid_search_n[3] = config.additive_scales

    # -------------------------------------------------------------------------------------------------------------
    # Covariance kernel
    # -------------------------------------------------------------------------------------------------------------

    model.print_title('Base Kernel Initialization')

    kernel = knp.Kernel(config.kernel_function,
                        parameters = kernel_parameter_grid_search_n,
                        base_kernel_without_BC = config.base_kernel_without_BC)

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
                                    mode_precision = 1e-14,
                                    testing_grid = X_test_decomposition,
                                    log_to_file = False,
                                    visualize = False )

    # -------------------------------------------------------------------------------------------------------------
    # GPR Configuration
    # -------------------------------------------------------------------------------------------------------------

    GP = gp.GPR(kernel, use_nugget = 'always', nugget_type = 'standard', nugget = nugget_val )


    # -------------------------------------------------------------------------------------------------------------
    # Perform CV
    # -------------------------------------------------------------------------------------------------------------

    CV_loss_0 = np.zeros((k_folds))
    CV_loss_1 = np.zeros((k_folds))

    for it_k in range(k_folds) :

        model.print_title(f'CV Fold {int(it_k + 1)} of {int(k_folds)}')

        fold_it = 'fold_' + str(it_k)

        # set CV test
        X_test = X_obs_for_CV[idx_folds[it_k],:]
        u_test_truth = values_obs_for_CV[idx_folds[it_k],:]

        # set observations
        observations_dict_k = {
            'structure' : 'velocity',
            'points'    : np.delete(X_obs_for_CV, idx_folds[it_k], axis=0),
            'values'    : np.delete(values_obs_for_CV, idx_folds[it_k], axis=0).flatten()
        }
        model.Gram_size = observations_dict_k['points'].shape[0]*2


        # -------------------------------------------------------------------------------------------------------------
        # GPR-based Reconstruction
        # -------------------------------------------------------------------------------------------------------------

        # Perform estimations
        out_dict = model.perform_GPR(GP, observations_dict_k, out_list = ['velocity'], X_particular = X_test)

        out_dict_check = model.perform_GPR(GP, observations_dict_k, out_list = ['obstacle', 'stream_obstacle'])
        abs_scalar_stream =np.abs(out_dict_check['stream_obstacle'])

        # compute profile boundary indicator (relative aggregated error on compact set)
        rel_agg_obstacle_normal, rel_agg_obstacle_tangent = model.compute_obstacle_normal_error(out_dict_check['obstacle'], relative = True)
        print(f'[Model] Relative-agg. normal components of velocity around obstacle  : {rel_agg_obstacle_normal}')


        # -------------------------------------------------------------------------------------------------------------
        # Compute and plot residuals
        # -------------------------------------------------------------------------------------------------------------

        u_residual = out_dict['velocity_particular'] - u_test_truth

        model.print_title('UQ computation')
        u_var = GP.interpolation('velocity_covariance', X_test, observations_dict_k)
        var_trace = np.diag(u_var)[::2] + np.diag(u_var)[1::2]
        total_SD = np.sqrt( var_trace )

        # -------------------------------------------------------------------------------------------------------------
        # Prediction interval coverage
        # -------------------------------------------------------------------------------------------------------------

        for i in range(2) :

            u_coverage_i = np.abs(u_residual[:,i]) / (1.96*np.sqrt(np.diag(u_var)[i::2]))
            fraction_i = np.sum(u_coverage_i <= 1.0)/ u_coverage_i.shape[0]
            print(f'Velocity component {i+1} coverage: {fraction_i}')

            if i == 0 :
                CV_loss_0[it_k] = fraction_i
            else :
                CV_loss_1[it_k] = fraction_i


    p0_loss_gs[it_gs] = CV_loss_0
    p1_loss_gs[it_gs] = CV_loss_1

    print(f'CV loss 0 mean: {p0_loss_gs[it_gs].mean()}')
    print(f'CV loss 1 mean: {p1_loss_gs[it_gs].mean()}')

# -------------------------------------------------------------------------------------------------------------
# Display CV loss
# -------------------------------------------------------------------------------------------------------------

def plot_loss(loss_plot, save_name) :

    loss_plot = np.flip(loss_plot.reshape((N_lcor, N_sigma)), axis = 0)
    val_max = loss_plot.max()
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(loss_plot, cmap="viridis")
    plt.colorbar()
    for i in range(loss_plot.shape[0]):
        for j in range(loss_plot.shape[1]):
            if val_max > 0 :
                text_color = "white" if loss_plot[i,j]/val_max < 0.7  else "black"
                plot_item = f"{loss_plot[i,j]:.1e}"
            else :
                text_color = "white" if loss_plot[i,j] < -2  else "black"
                plot_item = f"{loss_plot[i,j]:.1f}"
            plt.text(j, i, plot_item, ha="center", va="center", color=text_color)
    ax.set_xticks(range(len(sigma_grid)), labels=sigma_grid)
    lcor_grid_plot = np.flip(lcor_grid)
    ax.set_yticks(range(len(lcor_grid_plot)), labels=lcor_grid_plot)
    plt.xlabel('Standard deviation')
    plt.ylabel('Correlation length')

    plt.savefig(r".\BCGP_figures\\" + 'CV_loss_' + save_name + '.pdf', bbox_inches='tight', pad_inches=0.01)


plot_loss( ((p0_loss_gs-0.95)**2).mean(axis = 1) , save_name = 'horizontal_comp')
plot_loss( ((p1_loss_gs-0.95)**2).mean(axis = 1) , save_name = 'vertical_comp')

CV_loss = ((p0_loss_gs-0.95)**2).mean(axis = 1)/2  +  ((p1_loss_gs-0.95)**2).mean(axis = 1)/2
plot_loss(np.log10(CV_loss), save_name = 'log_total_UQ')


plt.show()
