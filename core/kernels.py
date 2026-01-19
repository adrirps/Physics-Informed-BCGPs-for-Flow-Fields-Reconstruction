
import numpy as np
import sympy as smp
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.path as pth
import os
import sys
import time
from datetime import datetime
from .profiles import NACA_airfoil


class Kernel(object):

    def __init__(self, kernel_name, **kwargs):

        self.scalar_kernel_name = kernel_name
        self.kernel_parameters = kwargs.get('parameters')

        if not kwargs.get('base_kernel_without_BC') is None :
            self.base_kernel_without_BC = kwargs.get('base_kernel_without_BC')
        else :
            self.base_kernel_without_BC = False

        # base kernel definition

        x1,x2,y1,y2 = smp.symbols('x1,x2,y1,y2')

        if kernel_name == 'RBF_anisotropic' :
            kappa = self.kernel_parameters[0]
            sigma_1 = self.kernel_parameters[1]
            sigma_2 = self.kernel_parameters[2]
            scalar_kernel = kappa * smp.exp( - ( ((x1-y1)/sigma_1)**2 + ((x2-y2)/sigma_2)**2 )/2 )

        elif kernel_name == 'RBF_anisotropic_additive' :
            sig  = self.kernel_parameters[0]
            lcor = self.kernel_parameters[1]
            alpha = self.kernel_parameters[2] # anisotropy at coarse scale 
            modes = self.kernel_parameters[3]

            gamma = 6 # 2D power-law

            scalar_kernel = ((sig)**2) * smp.exp( - ( ((x1-y1)**2 + (alpha*(x2-y2))**2) / (lcor**2) ) /2 )
            for m in range(1, int(modes) + 1) :
                alpha_m = max(alpha/ (2**(m)), 1.0)
                print(f'[Kernel] Anisotropy parameter for fine scales (m = {m}): {alpha_m}')

                scalar_kernel += ((sig/(2**(gamma*m)))**2) * smp.exp( - ( ((x1-y1)**2 + (alpha_m*(x2-y2))**2) / ((lcor/(2**(3*m)))**2) ) /2 )

        self.scalar_kernel_np = smp.lambdify([x1,x2,y1,y2], scalar_kernel, 'numpy')
        self.scalar_kernel = scalar_kernel

        print(f'[Kernel] Scalar kernel <<{kernel_name}>> is defined !')


    def load_derivatives(self) :

        print('[Kernel] Loading derivatives...')

        x1,x2,y1,y2 = smp.symbols('x1,x2,y1,y2')

        # Dx1
        Dx1_G = smp.diff(self.scalar_kernel,x1)
        # Dx2
        Dx2_G = smp.diff(self.scalar_kernel,x2)
        # Dy1
        Dy1_G = smp.diff(self.scalar_kernel,y1)
        # Dy2
        Dy2_G = smp.diff(self.scalar_kernel,y2)

        # Dx1_Dy1
        Dx1_Dy1_G = smp.diff(self.scalar_kernel,x1,y1)
        # Dx1_Dy2
        Dx1_Dy2_G = smp.diff(self.scalar_kernel,x1,y2)
        # Dx2_Dy1
        Dx2_Dy1_G = smp.diff(self.scalar_kernel,x2,y1)
        # Dx2_Dy2
        Dx2_Dy2_G = smp.diff(self.scalar_kernel,x2,y2)
        # DDx1
        DDx1_G = smp.diff(self.scalar_kernel,x1,2)
        # DDx2
        DDx2_G = smp.diff(self.scalar_kernel,x2,2)
        # Lx
        Lx_G = DDx1_G + DDx2_G
        # DDy1
        DDy1_G = smp.diff(self.scalar_kernel,y1,2)
        # DDy2
        DDy2_G = smp.diff(self.scalar_kernel,y2,2)
        # Ly
        Ly_G = DDy1_G + DDy2_G

        Dx1_Lx_G = smp.diff(Lx_G,x1)
        # Dx2_Lx
        Dx2_Lx_G = smp.diff(Lx_G,x2)

        # Dx1_Ly
        Dx1_Ly_G = smp.diff(Ly_G,x1)
        # Dx2_Ly
        Dx2_Ly_G = smp.diff(Ly_G,x2)

        # Dy1_Lx
        Dy1_Lx_G = smp.diff(Lx_G,y1)
        Lx_Dy1_G = Dy1_Lx_G
        # Dy2_Lx
        Dy2_Lx_G = smp.diff(Lx_G,y2)
        Lx_Dy2_G = Dy2_Lx_G

        # Order 4

        # D4x1 (or DDDDx1)
        D4x1_G = smp.diff(self.scalar_kernel,x1,4)
        # D4x2 (or DDDDx1)
        D4x2_G = smp.diff(self.scalar_kernel,x2,4)
        # DDx1_DDx2
        DDx1_DDx2_G = smp.diff(DDx1_G,x2,2)

        # LxLx
        LxLx_G = D4x1_G + (2 * DDx1_DDx2_G) + D4x2_G

        # DDx1_DDy1
        DDx1_DDy1_G = smp.diff(DDy1_G,x1,2)
        # DDx1_DDy2
        DDx1_DDy2_G = smp.diff(DDy2_G,x1,2)
        # DDx2_DDy1
        DDx2_DDy1_G = smp.diff(DDy1_G,x2,2)
        # DDx2_DDy2
        DDx2_DDy2_G = smp.diff(DDy2_G,x2,2)

        # LxLy
        LxLy_G = DDx1_DDy1_G + DDx1_DDy2_G + DDx2_DDy1_G + DDx2_DDy2_G

        # Dx1_Lx_Dy1
        Dx1_Lx_Dy1_G = smp.diff(Lx_Dy1_G,x1) 
        # Dx1_Lx_Dy2
        Dx1_Lx_Dy2_G = smp.diff(Lx_Dy2_G,x1) 
        # Dx2_Lx_Dy1
        Dx2_Lx_Dy1_G = smp.diff(Lx_Dy1_G,x2) 
        # Dx2_Lx_Dy2
        Dx2_Lx_Dy2_G = smp.diff(Lx_Dy2_G,x2) 

        # Order 5

         # Dy1_LxLx
        Dy1_LxLx_G = smp.diff(LxLx_G,y1,1)
         # Dy2_LxLx
        Dy2_LxLx_G = smp.diff(LxLx_G,y2,1)

        # # Dx1_LxLy
        Dx1_LxLy_G = smp.diff(LxLy_G,x1,1)
        # # Dx2_LxLy
        Dx2_LxLy_G = smp.diff(LxLy_G,x2,1)

        # DDx1_LxLy
        DDx1_LxLy_G = smp.diff(LxLy_G,x1,2)
        # DDx2_LxLy
        DDx2_LxLy_G = smp.diff(LxLy_G,x2,2)
        # LxLxLy
        LxLxLy_G = DDx1_LxLy_G + DDx2_LxLy_G

        self.derivatives_dict_smp = {
            ('id', 'id'): self.scalar_kernel, # Order 0
            ('Dx1','id') : Dx1_G, # Order 1
            ('Dx2','id') : Dx2_G,
            ('id','Dy1') : Dy1_G,
            ('id','Dy2') : Dy2_G,
            ('Dx1','Dy1') : Dx1_Dy1_G, # Order 2
            ('Dx1','Dy2') : Dx1_Dy2_G,
            ('Dx2','Dy1') : Dx2_Dy1_G,
            ('Dx2','Dy2') : Dx2_Dy2_G,
            ('Lx','id')  : Lx_G,
            ('id','Ly')  : Ly_G,
            ('Dx1_Lx','id')  : Dx1_Lx_G, # Order 3
            ('Dx2_Lx','id')  : Dx2_Lx_G,
            ('Lx','Dy1') : Lx_Dy1_G,
            ('Lx','Dy2') : Lx_Dy2_G,
            ('Dx1','Ly') : Dx1_Ly_G,
            ('Dx2','Ly') : Dx2_Ly_G,
            ('Dx1_Lx', 'Dy1') : Dx1_Lx_Dy1_G, # Order 4
            ('Dx1_Lx', 'Dy2') : Dx1_Lx_Dy2_G,
            ('Dx2_Lx', 'Dy1') : Dx2_Lx_Dy1_G,
            ('Dx2_Lx', 'Dy2') : Dx2_Lx_Dy2_G,
            ('Lx', 'Ly') : LxLy_G,
            ('Lx_Lx' , 'id') : LxLx_G,
            ('Lx_Lx' , 'Dy1') : Dy1_LxLx_G, # Order 5
            ('Lx_Lx' , 'Dy2') : Dy2_LxLx_G,
            ('Dx1_Lx', 'Ly') : Dx1_LxLy_G,
            ('Dx2_Lx', 'Ly') : Dx2_LxLy_G,
            ('Lx_Lx' , 'Ly') : LxLxLy_G, # Order 6
        }

        # create numpy functions from sympy kernel derivatives
        derivatives_dict_np = {}
        for key in self.derivatives_dict_smp :
            derivatives_dict_np[key] = smp.lambdify([x1,x2,y1,y2], self.derivatives_dict_smp[key], 'numpy')
        self.derivatives_dict_np = derivatives_dict_np

        print('[Kernel] Kernel derivatives are computed !')


    def obstacle_spectral_decomposition(self, obstacle_type, obstacle_parameters, **kwargs) :

        # saving configuration

        self.start_BC_time = time.time()

        self.obstacle_type = obstacle_type
        self.obstacle_parameters = obstacle_parameters
        
        self.visualize = kwargs.get('visualize')
        
        # save log to file (optional)

        if 'log_to_file' in kwargs and kwargs.get('log_to_file') :
            date_time_label = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_kernel_folder_path = kwargs.get('kernels_folder_path') + '\\kernel_' + self.scalar_kernel_name + '_' + date_time_label + '\\'
            os.makedirs(os.path.dirname(new_kernel_folder_path), exist_ok = True)

            log_file_path = new_kernel_folder_path + 'BC_log.txt'
            self.log_file = open(log_file_path, "w")
            sys.stdout = self.log_file
            print('[Kernel] Start of log file')
            print(f'[Kernel] Defined kernel parameters : {self.kernel_parameters}')
            self.log_to_file = True

        # obstacle domain

        if 'kernel_obstacle_domain' in kwargs :
            self.obstacle_limits = kwargs.get('kernel_obstacle_domain')
        else :
            self.obstacle_limits = [0,2*np.pi]

        print(f'[Kernel] Obstacle limits for decomposition set to [{self.obstacle_limits[0]:.1f}, {self.obstacle_limits[1]:.1f}]')


        if 'testing_grid' in kwargs :
            testing_grid = kwargs.get('testing_grid')

        N_integration = kwargs.get('N_integration') 
        mode_precision = kwargs.get('mode_precision')
        KL_measure = kwargs.get('KL_measure')
        self.eigen_decomposition(KL_measure = KL_measure,
                                N_integration = N_integration, mode_precision = mode_precision,
                                test_eigenfunction_accuracy = True,
                                test_decomposition_compact_set = True,
                                test_decomposition_compact_set_grid = testing_grid,
                                test_derivatives_decomposition = True)
        
        # precompute kernel derivatives
        self.precompute_kernel_derivatives_with_decomposition()

        self.end_BC_time = time.time()

        print(f'[Kernel] BC procedure duration : {(self.end_BC_time - self.start_BC_time)/60:.4f} minutes ')
        print(f'[Kernel] BC finish date time : {datetime.now().strftime('%Y%m%d_%H%M%S')} ')

        # end log file
        if hasattr(self, 'log_to_file') and self.log_to_file :
            print('[Kernel] End of log file')
            sys.stdout = sys.__stdout__
            self.log_file.close()


    def eigen_decomposition(self, KL_measure, N_integration, mode_precision, **kwargs):

        # setting parameter

        if 'N_test' in kwargs :
            N_test = kwargs.get('N_test')
            
        if self.obstacle_type == 'cylinder' : 

            # set transformation
            s,t = smp.symbols('s,t')

            x0 = self.obstacle_parameters[0]
            y0 = self.obstacle_parameters[1]
            if self.obstacle_type == 'cylinder':
                radius = self.obstacle_parameters[2]
                self.radius = radius
                self.gamma_1_s = x0 + radius*smp.cos(s)
                self.gamma_2_s = y0 + radius*smp.sin(s)
                gamma_1_t = x0 + radius*smp.cos(t)
                gamma_2_t = y0 + radius*smp.sin(t)
            elif self.obstacle_type == 'ellipse':
                a = self.obstacle_parameters[2]
                b = self.obstacle_parameters[3]
                self.gamma_1_s = x0 + a*smp.cos(s)
                self.gamma_2_s = y0 + b*smp.sin(s)
                gamma_1_t = x0 + a*smp.cos(t)
                gamma_2_t = y0 + b*smp.sin(t)

            self.gamma_1_np = smp.lambdify([s], self.gamma_1_s, 'numpy')
            self.gamma_2_np = smp.lambdify([s], self.gamma_2_s, 'numpy')

            s_grid_obstacle = np.linspace(0, 2*np.pi, N_integration) # all integration nodes
            X_obstacle = np.zeros((N_integration,2))
            X_obstacle[:,0] = self.gamma_1_np(s_grid_obstacle)
            X_obstacle[:,1] = self.gamma_2_np(s_grid_obstacle)

        
        elif self.obstacle_type == 'NACA_airfoil' :
            center_0 = self.obstacle_parameters[0]
            center_1 = self.obstacle_parameters[1]
            NACA_code = self.obstacle_parameters[2]
            scale_length = self.obstacle_parameters[3]
            self.airfoil = NACA_airfoil(NACA_code)
            X_obstacle, s_grid = self.airfoil.boundary_values(N_integration, scale_length, limits = self.obstacle_limits)
            X_obstacle[:,0] = X_obstacle[:,0] + center_0
            X_obstacle[:,1] = X_obstacle[:,1] + center_1
            if self.visualize :
                self.airfoil.plot_airfoil(X_obstacle[:,0],X_obstacle[:,1])

            # compute transformation gradient (for KL measure)
            gamma_tangent_grid = self.airfoil.curve_derivatives_values(1, scale_length, s_series = s_grid,
                                                                                    normalized = False,
                                                                                    not_filter = True )

        # save obstacle points
        self.X_obstacle = X_obstacle

        # set discretization of eigenproblem
        N_eigenprob = N_integration - 1 # Eigenprob integration nodes = N_eigenprob + 1
        t_series_eigenprob = np.linspace(self.obstacle_limits[0],self.obstacle_limits[1],N_eigenprob + 1)
        
        # compute matrix
        A = np.zeros((N_eigenprob,N_eigenprob)) # Gram matrix

        # rectangle rule in eigen problem integral
        X_obstacle = X_obstacle[:-1,]

        A_base = self.compute_kernel('id','id', X_obstacle, X_obstacle, base = True)

        h_eigen = (self.obstacle_limits[1]-self.obstacle_limits[0])/N_eigenprob

        if KL_measure == 'pushforward' :

            proba_normalization_mesaure = self.obstacle_limits[1] - self.obstacle_limits[0]

            sqrt_H = np.zeros(A_base.shape)
            np.fill_diagonal(sqrt_H, np.sqrt(h_eigen / proba_normalization_mesaure)) # rectangle rule

        elif KL_measure == 'surface' :

            # rectangle rule in eigenproblem integral
            gamma_tangent_grid = gamma_tangent_grid[:-1,]

            gamma_tangent_norms = np.linalg.norm(gamma_tangent_grid, axis = 1)
            proba_normalization_mesaure = np.sum(gamma_tangent_norms*h_eigen)

            sqrt_H = np.diag(np.sqrt(gamma_tangent_norms / proba_normalization_mesaure))

        else :
            raise ValueError('[Kernel] Measure for KL expasion is not defined !')

        A = np.matmul(sqrt_H,(np.matmul(A_base,sqrt_H)))
        
        # solve eigenvalue problem
        Lambda, discrete_eigenfunctions = np.linalg.eigh(A) # columns are eigefunction vectors

        # eigenvalue error
        eigen_matrix_error = np.abs(np.matmul(A,discrete_eigenfunctions) - np.matmul(discrete_eigenfunctions,np.diag(Lambda)))
        print(f'[Kernel] Eigen matrix error max : {eigen_matrix_error.max()}')
        print(f'[Kernel] Eigen matrix error mean : {eigen_matrix_error.mean()}')

        # sort descending order
        descending_indexes = np.argsort(-Lambda)
        Lambda = Lambda[descending_indexes]
        discrete_eigenfunctions = discrete_eigenfunctions[:,descending_indexes]

        # set number of eigenvalues from precision
        N_eigen = 1
        trace_A = np.trace(A)
        trace_indicator = np.sum(Lambda[0:N_eigen]) / trace_A
        while( trace_indicator <= 1 - mode_precision) :
            if Lambda[N_eigen-1] < 1e-17 : # computation precision
                print(f'[Kernel] Eigenvalue and fixed mode precision are too small, stopping indicator test')
                break
            N_eigen += 1
            trace_indicator = np.sum(Lambda[0:N_eigen]) / trace_A
        
        N_eigen_limit = 200 # upper limit
        if N_eigen > N_eigen_limit :
            N_eigen = N_eigen_limit
            trace_indicator = np.sum(Lambda[0:N_eigen]) / trace_A
            print(f'[Kernel] Precision is too small. Setting number of modes to limit {N_eigen_limit}')
            print(f'[Kernel] Best precision is : {1 - trace_indicator:.5e}')

        print(f'[Kernel] Indicator based on trace (sum of modes over trace) = {trace_indicator} for precision {mode_precision}')
        print(f'[Kernel] Number of modes used N_eigen = {N_eigen}')
        print(f'[Kernel] Last (smallest) eigenvalue to be used = {Lambda[N_eigen-1]}')

        if 'eigenvalue_truncation' in kwargs and kwargs.get('eigenvalue_truncation') :
            self.N_eigen_truncation = N_eigen
            print(f'[Kernel] Eigenvalue truncated at N_eigen = {N_eigen}....')
            return 1 - trace_indicator

        # Eigenfunction L2(compact set) normalization
        # weigth already in KL measure
        measure_eigenfun_norms = np.diag(np.matmul(discrete_eigenfunctions.transpose(),discrete_eigenfunctions))

        for it_n in range(N_eigen) :
            discrete_eigenfunctions[:,it_n] = discrete_eigenfunctions[:,it_n] / measure_eigenfun_norms[it_n]

        # Spectral factor for computations

        weighted_eigenfunctions = np.matmul(sqrt_H, discrete_eigenfunctions) # for spectral integral computation

        # Save decomposition elements

        self.Lambda = Lambda[0:N_eigen]

        self.Lambda_all = Lambda

        self.discrete_eigenfunctions = discrete_eigenfunctions # all eigenfunctions version

        self.weighted_eigenfunctions = weighted_eigenfunctions # all eigenfunctions version

        self.KL_measure = KL_measure

        self.N_integration = N_integration # for computing derivatives reconstruction

        self.mode_precision = mode_precision # for saving eigenelements

        # Eigenvalues plot

        plt.figure()
        plt.plot(Lambda[:N_eigen], '-o')
        plt.title(r'Retained eigenvalues of Karhunen-Loève Expansion')

        if Lambda[N_eigen-1] <= 0 :
            raise ValueError("Retained eigenvalues cannot be zero or negative")
        print(f'[Kernel] Last eigenvalue to be used ({N_eigen}-th) : {Lambda[N_eigen-1]}')
        print(f'[Kernel] Next to last eigenvalue to be used ({N_eigen+1}-th) : {Lambda[N_eigen]}')


        # Test (1) : decomposition on compact set (from discrete vector)

        if 'test_decomposition_compact_set' in kwargs and kwargs.get('test_decomposition_compact_set') == True :

            N_test = N_integration - 1

            print(f'[Kernel] Number of test for kernel decomposition = {N_test}')

            temp_G = self.compute_kernel('id','id',X_obstacle, X_obstacle, base = True)

            avoid_first_point = False

            try :
                inv_sqrt_H = np.linalg.inv(sqrt_H)
                discrete_eigenfunctions_temp = discrete_eigenfunctions

            except :
                # avoid h_0 = 0 (in case of NACA definition)
                avoid_first_point = True
                discrete_eigenfunctions_temp = discrete_eigenfunctions[1:,:] 
                inv_sqrt_H = np.linalg.inv(sqrt_H[1:,1:])
                temp_G = temp_G[1:,1:] # for comparison

            original_eigenfunctions = np.matmul(inv_sqrt_H, discrete_eigenfunctions_temp)

            N_test = min(N_integration - 2, 100)

            temp_G = temp_G[:N_test,:N_test]

            decomposition_temp = np.zeros((N_test,N_test))
            for i_t in range(N_test) :
                for j_t in range(N_test) :
                    decomposition_temp_ij = 0
                    for n in range(N_eigen) :
                        decomposition_temp_ij += Lambda[n]*original_eigenfunctions[i_t,n]*original_eigenfunctions[j_t,n]
                    decomposition_temp[i_t,j_t] = decomposition_temp_ij

            test_decomposition_error =  temp_G - decomposition_temp

            print(f'[Kernel] G Mercer decomposition on compact_set order (mean) : {np.abs(temp_G).mean()}')
            print(f'[Kernel] G Mercer decomposition on compact_set error min : {np.abs(test_decomposition_error).min()}')
            print(f'[Kernel] G Mercer decomposition on compact_set error mean : {np.abs(test_decomposition_error).mean()}')
            print(f'[Kernel] G Mercer decomposition on compact_set error max : {np.abs(test_decomposition_error).max()}')


        # Test (2) : eigenfunction accuracy (point-wise)

        if 'test_eigenfunction_accuracy' in kwargs and kwargs.get('test_eigenfunction_accuracy') == True :

            weighted_eigenfunctions = np.matmul(sqrt_H,discrete_eigenfunctions)

            weighted_eigenfunctions_trunc = weighted_eigenfunctions[:,0:N_eigen]
            spectral_factor = np.matmul(weighted_eigenfunctions_trunc, np.diag(1/np.sqrt(self.Lambda)) )
            temp_G_base = self.compute_kernel('id','id', X_obstacle, X_obstacle, base = True)

            test_integral_array = np.matmul(temp_G_base, spectral_factor)

            if avoid_first_point : # from Test (1)
                test_integral_array = test_integral_array[1:,:] # avoid h_0 = 0 (in case of NACA definition)

            test_eigenfun_array = np.matmul(original_eigenfunctions[:,0:N_eigen], np.diag(np.sqrt(self.Lambda))) # from Test (1)

            test_eigenfunctions = test_eigenfun_array - test_integral_array

            print(f'[Kernel] Eigenfunction order : {np.abs(discrete_eigenfunctions).mean()}')
            print(f'[Kernel] Eigenfunction error min : {np.abs(test_eigenfunctions).min()}')
            print(f'[Kernel] Eigenfunction error mean : {np.abs(test_eigenfunctions).mean()}')
            print(f'[Kernel] Eigenfunction error max : {np.abs(test_eigenfunctions).max()}')

        # Test (3) : decomposition on compact set (from eigenfunction integral reconstruction)

        if 'test_decomposition_compact_set' in kwargs and kwargs.get('test_decomposition_compact_set') == True :

            # Grid for testing
            if 'test_decomposition_compact_set_grid' in kwargs : 
                X_test_decomposition = kwargs.get('test_decomposition_compact_set_grid')
                N_test = X_test_decomposition.shape[0]

            weighted_eigenfunctions = np.matmul(sqrt_H,discrete_eigenfunctions)

            weighted_eigenfunctions_trunc = weighted_eigenfunctions[:,0:N_eigen]
            spectral_factor = np.matmul(weighted_eigenfunctions_trunc, np.diag(1/np.sqrt(self.Lambda)) )

            # compute eigenfunction reconstruction test grid
            temp_G_base_test = self.compute_kernel('id','id', X_test_decomposition, X_obstacle, base = True)
            extension_test_int = np.matmul(temp_G_base_test, spectral_factor)

            # compute eigenfunction reconstruction obstacle grid
            temp_G_base_obstacle = self.compute_kernel('id','id', X_obstacle, X_obstacle, base = True)
            extension_obstacle_int = np.matmul(temp_G_base_obstacle, spectral_factor)

            decomposition_temp = np.matmul(extension_test_int,extension_obstacle_int.T)

            temp_G = self.compute_kernel('id','id', X_test_decomposition, X_obstacle, base = True)
            test_decomposition_error =  temp_G - decomposition_temp
            mean_test_decomposition_error = np.abs(test_decomposition_error).mean(axis = 1)

            print(f'[Kernel] G integral decomposition (on D) order : {np.abs(temp_G).mean()}')
            print(f'[Kernel] G integral decomposition (on D) error min : {mean_test_decomposition_error.min()}')
            print(f'[Kernel] G integral decomposition (on D) error mean : {mean_test_decomposition_error.mean()}')
            print(f'[Kernel] G integral decomposition (on D) error max : {mean_test_decomposition_error.max()}')

            # Plot
            plt.figure(figsize=(10,4))
            plt.plot(X_obstacle[:,0],X_obstacle[:,1])
            plt.scatter(X_test_decomposition[:,0],X_test_decomposition[:,1], c = mean_test_decomposition_error)
            cbar = plt.colorbar(format="%.1e")
            cbar.set_label('Mean of evaluations over obstacle', rotation = 270, labelpad=15)
            plt.axis('equal')
            plt.title(r'Approximation of $G(\cdot,x^\prime)= 0 $ over obstacle, for different choice of $x^\prime$')

            print('[Kernel] Kernel Eigenproblem finished !')


    def precompute_kernel_derivatives_with_decomposition(self) :

        N_eigen = self.Lambda.shape[0]
        self.spectral_factor = np.matmul(self.weighted_eigenfunctions[:,0:N_eigen], np.diag(1/np.sqrt(self.Lambda_all[0:N_eigen])) )

        print('[BCGP Kernel] Spectral factor pre-computed for kernel derivatives ! ')
    
        return


    def compute_kernel(self, label_x, label_y, X_grid, Y_grid, **kwargs) :

        x1 = X_grid[:,0]
        x2 = X_grid[:,1]
        y1 = Y_grid[:,0]
        y2 = Y_grid[:,1]

        # compute base kernel values
        N_x = x1.shape[0]
        N_y = y1.shape[0]
        XX1 = np.tile(x1,(N_y,1)).transpose()
        XX2 = np.tile(x2,(N_y,1)).transpose()
        YY1 = np.tile(y1,(N_x,1))
        YY2 = np.tile(y2,(N_x,1))

        base_kernel_np = self.derivatives_dict_np[(label_x,label_y)]
        
        if hasattr(self,'is_MLE_kernel') and self.is_MLE_kernel :
            hyparam = kwargs.get('kernel_hyperparams')
            base_val = base_kernel_np(XX1,XX2,YY1,YY2, *hyparam)
        else :
            base_val = base_kernel_np(XX1,XX2,YY1,YY2)

        if self.base_kernel_without_BC or kwargs.get('base') :

            return base_val

        # compute BCGP kernel

        left_kernel_term = self.compute_kernel( label_x, 'id', X_grid, self.X_obstacle[:-1,:], base = True)
        integral_operator_left_val = np.matmul(left_kernel_term, self.spectral_factor)

        label_y_eigen = label_y.replace('y', 'x')
        right_kernel_term = self.compute_kernel(label_y_eigen, 'id', Y_grid, self.X_obstacle[:-1,:], base = True)
        integral_operator_right_val = np.matmul(right_kernel_term, self.spectral_factor).transpose()

        if hasattr(self,'N_eigen_truncation') and self.N_eigen_truncation < integral_operator_right_val.shape[0] :
            extension_val = np.matmul(integral_operator_left_val[:,:self.N_eigen_truncation], integral_operator_right_val[:self.N_eigen_truncation,:])
            print(f'[Kernel] BCGP computations using truncation at {self.N_eigen_truncation} modes')
        else :
            extension_val = np.matmul(integral_operator_left_val, integral_operator_right_val)

        return base_val - extension_val


    def compute_matrix_incompressible(self, label_x, label_y, X, Y, **kwargs) :

        if  label_x == 'id' and label_y == 'id' :

            # kernel computations
            K11 = self.compute_kernel('Dx2','Dy2', X, Y, **kwargs)
            K12 = - self.compute_kernel('Dx2','Dy1', X, Y, **kwargs)
            K21 = - self.compute_kernel('Dx1','Dy2', X, Y, **kwargs)
            K22 = self.compute_kernel('Dx1','Dy1', X, Y, **kwargs)

            N_val_x = X.shape[0] * 2
            N_val_y = Y.shape[0] * 2
            val = np.zeros((N_val_x, N_val_y))

            # assembly
            val[0::2, 0::2] = K11  # (0,0) sub-position
            val[0::2, 1::2] = K12  # (0,1) sub-position
            val[1::2, 0::2] = K21  # (1,0) sub-position
            val[1::2, 1::2] = K22  # (1,1) sub-position

        return val
    
    def compute_vector_stream(self, X, Y) :

        # kernel computations
        P1 = - self.compute_kernel('id','Dy2', X, Y)
        P2 = self.compute_kernel('id','Dy1', X, Y)

        N_val_x = X.shape[0]
        N_val_y = Y.shape[0] * 2
        val = np.zeros((N_val_x, N_val_y))

        # assembly
        val[:, 0::2] = P1  # (0,0) sub-position
        val[:, 1::2] = P2  # (0,1) sub-position

        return val


    def set_mode_truncation(self, precision, KL_measure, N_integration) :
        spectral_precision = self.eigen_decomposition(KL_measure = KL_measure,
                                                      N_integration = N_integration,
                                                      mode_precision = precision,
                                                      eigenvalue_truncation = True )
        
        return spectral_precision
        
    def create_testing_grid(self, domain, obstacle_type, obstacle_parameters, N_test_obstacle, N_test_domain) :

        # obstacle parameters for generic grid construction (outside BC procedure)
        # Spectral method test

        self.obstacle_type = obstacle_type

        X_test_domain_pre = self.build_grid(N_test_domain, domain, with_limits = True)

        if obstacle_type == 'NACA_airfoil' :
            NACA_code = obstacle_parameters[2]
            chord_length = obstacle_parameters[3]
            airfoil = NACA_airfoil(NACA_code)
            X_test_decomposition_obstacle, temp_grid = airfoil.boundary_values(N_test_obstacle, chord_length, limits = [0,2*np.pi-0.07127]) # shift for different obstacle points
            X_test_decomposition_obstacle = X_test_decomposition_obstacle

            X_obstacle_for_filter, temp_grid = airfoil.boundary_values(5000, chord_length, limits = [0,2*np.pi])
            X_obstacle_for_filter = X_obstacle_for_filter[:-1,:] # closed set
        
        elif obstacle_type == 'cylinder' :
            center_x = obstacle_parameters[0]
            center_y = obstacle_parameters[1]
            radius = obstacle_parameters[2]
            
            # testing grid obstacle
            s_grid_test = np.linspace(0, 2*np.pi-0.07127, N_test_obstacle) # shift for different obstacle points
            X_test_decomposition_obstacle = np.zeros((N_test_obstacle,2))
            X_test_decomposition_obstacle[:,0] =  center_x + radius*np.cos(s_grid_test)
            X_test_decomposition_obstacle[:,1] =  center_y + radius*np.sin(s_grid_test)

            # testing grid domain
            s_grid_filter = np.linspace(0, 2*np.pi, 5000 + 1)[:-1] # closed set
            X_obstacle_for_filter = np.zeros((5000, 2))
            X_obstacle_for_filter[:,0] =  center_x + radius*np.cos(s_grid_filter)
            X_obstacle_for_filter[:,1] =  center_y + radius*np.sin(s_grid_filter)

        X_test_domain, temp_indexes = self.filter_domain_obstacle(X_test_domain_pre, X_obstacle = X_obstacle_for_filter)
        X_test_decomposition = np.vstack((X_test_decomposition_obstacle, X_test_domain))

        return X_test_decomposition


    def build_grid(self, N_grid, domain, **kwargs) :

        domain = np.array(domain)
        ratio_x = (domain[0,1] - domain[0,0]) / (domain[1,1] - domain[1,0])
        N_unit_float = np.sqrt(N_grid/ratio_x)
        N_x = int(ratio_x*N_unit_float)
        N_y = int(np.round(N_unit_float))

        if kwargs.get('with_limits') :
            X = np.linspace(domain[0,0], domain[0,1], N_x )
            Y = np.linspace(domain[1,0], domain[1,1], N_y )
        else :
            X = np.linspace(domain[0,0], domain[0,1], N_x +2 )[1:-1]
            Y = np.linspace(domain[1,0], domain[1,1], N_y +2 )[1:-1]

        XX, YY = np.meshgrid(X,Y)
        N_grid = XX.flatten().shape[0]
        X_grid = np.zeros((N_grid,2))
        X_grid[:,0] = XX.flatten()
        X_grid[:,1] = YY.flatten()

        if 'N_x_out' in kwargs :
            return X_grid, N_x

        return X_grid
    
    def filter_domain_obstacle(self, X_domain, X_obstacle, **kwargs):
        
        obstacle_path = pth.Path(X_obstacle)
        inner_indexes = obstacle_path.contains_points(X_domain)
        indexes = ~ inner_indexes

        X_domain = X_domain[indexes,:]

        print('[Points] Domain points filtered according to obstacle boundary constraint')

        if 'inner' in kwargs and kwargs.get('inner') :
            return X_domain, indexes, inner_indexes
        else : 
            return X_domain, indexes


    def filter_close_points(self, X_grid, tol_dist, **kwargs) :
        X_grid_tree = sp.spatial.KDTree(X_grid)
        indexes = np.ones(len(X_grid), dtype=bool)

        for i_grid in range(len(X_grid)):
            if not indexes[i_grid]:  
                continue
            neighbors = X_grid_tree.query_ball_point(X_grid[i_grid], tol_dist) # to remove
            for it_neighbors in neighbors:
                if it_neighbors > i_grid:
                    if 'airfoil' in kwargs and kwargs.get('airfoil') :
                        if i_grid < int(len(X_grid)/4) and it_neighbors > int(len(X_grid)/2) :
                            pass # keep
                        elif i_grid > len(X_grid) - int(len(X_grid)/4) and it_neighbors < int(len(X_grid)/4) :
                            pass
                        else :
                            indexes[it_neighbors] = False
                    else :
                        indexes[it_neighbors] = False

        
        return X_grid[indexes], indexes
