
import numpy as np
import scipy as sp

class GPR() :

    def __init__(self, kernel_function, **kwargs):

        self.kernel_function = kernel_function
        self.set_nugget(**kwargs) # for regression function


    def set_nugget(self, **kwargs) :

        if ('use_nugget' in kwargs) and ('nugget_type' in kwargs) :
            self.use_nugget = kwargs.get('use_nugget')
            self.nugget_type = kwargs.get('nugget_type')
            self.nugget = kwargs.get('nugget')
            if self.use_nugget == 'always' :
                print(f'[GPR] Nugget {self.nugget} set for all computations')
            elif self.use_nugget == 'when_necessary' :
                print(f'[GPR] Nugget {self.nugget} set only when computation fails')


    def interpolation(self, formula, X_new, observations_dict) :

        obs_structure = observations_dict['structure']

        if 'points' in observations_dict :
            X_obs = observations_dict['points']
            values_cond = observations_dict['values']

        if obs_structure == 'velocity':

            # Posterior mean

            if formula.endswith('mean') :

                Cross_matrix = self.Cross_covariance_assembly(formula, obs_structure, X_new, observations_dict)
                Gram_matrix = self.kernel_function.compute_matrix_incompressible('id','id', X_obs, X_obs)

                val = self.regression(Cross_matrix, Gram_matrix, values_cond)

            # Posterior covariance
            elif formula == 'velocity_covariance' :

                Gram_matrix = self.kernel_function.compute_matrix_incompressible('id','id', X_obs, X_obs)

                if not isinstance(X_new, dict):

                    print(f'[GPR] Computing Cross variance with matrix kernel over {X_new.shape[0]} points')

                    Cross_matrix_left = self.kernel_function.compute_matrix_incompressible('id','id', X_new, X_obs)
                    Cross_matrix_right = Cross_matrix_left.transpose()

                    print(f'[GPR] Computing prior variance with matrix kernel over {X_new.shape[0]} points')
                    Cov_priori = self.kernel_function.compute_matrix_incompressible('id','id', X_new, X_new)

                    print(f'[GPR] Kernel computations done')

                else :

                    print(f'[GPR] Computing covariance with matrix kernel over {X_new['left'].shape[0]} x {X_new['right'].shape[0]} points')

                    Cross_matrix_left = self.kernel_function.compute_matrix_incompressible('id','id', X_new['left'], X_obs)
                    Cross_matrix_right = self.kernel_function.compute_matrix_incompressible('id','id', X_obs, X_new['right'])
                    Cov_priori = self.kernel_function.compute_matrix_incompressible('id','id', X_new['left'], X_new['right'])

                val_reg = self.regression(Cross_matrix_left, Gram_matrix, Cross_matrix_right)
                val = Cov_priori - val_reg

        elif obs_structure == 'velocity_and_normal' :

            # Posterior mean with observations from velocity and normal component ( curl u \cdot n = 0 )

            values_cond_normal  = np.zeros((observations_dict['points_normal'].shape[0],)) # values set to zero
            values_cond_all = np.concatenate((values_cond, values_cond_normal))

            print(f'[GPR] Computing mean with matrix kernel over {X_new.shape[0]} new points')

            Cross_matrix = self.Cross_covariance_assembly(formula, obs_structure, X_new, observations_dict)
            Gram_matrix = self.Gram_matrix_assembly(obs_structure, observations_dict)

            val = self.regression(Cross_matrix, Gram_matrix, values_cond_all)


        return val
    
    def Cross_covariance_assembly(self, formula, obs_structure, X_new, observations_dict):

        if obs_structure == 'velocity':

            # get positions
            X_obs = observations_dict['points']

            if formula == 'velocity_mean':

                Cross_matrix = self.kernel_function.compute_matrix_incompressible('id','id', X_new, X_obs)

            elif formula == 'scalar_stream_mean' :

                Cross_matrix = self.kernel_function.compute_vector_stream(X_new, X_obs)


        elif obs_structure == 'velocity_and_normal':

            # get positions
            X_obs = observations_dict['points']
            X_obs_normal = observations_dict['points_normal']
            Cross_dim = 2*X_obs.shape[0] + X_obs_normal.shape[0]

            # set normal vectors
            normal_vectors = observations_dict['normal_vectors']
            if normal_vectors.ndim > 1 :
                normal_vectors = normal_vectors.flatten()[:, np.newaxis]

            # computations
            if formula == 'velocity_mean':

                Cross_matrix = np.zeros((2*X_new.shape[0], Cross_dim))

                Cross_matrix[:,:2*X_obs.shape[0]] = self.kernel_function.compute_matrix_incompressible('id','id', X_new, X_obs)
                Cross_K_2 = self.kernel_function.compute_matrix_incompressible('id','id', X_new, X_obs_normal)

                normal_vectors = np.tile(normal_vectors.transpose(), (2*X_new.shape[0],1))

            elif formula == 'scalar_stream_mean':

                Cross_matrix = np.zeros((X_new.shape[0], Cross_dim))

                Cross_matrix[:,:2*X_obs.shape[0]] = self.kernel_function.compute_vector_stream(X_new, X_obs)
                Cross_K_2 = self.kernel_function.compute_vector_stream(X_new, X_obs_normal)

                normal_vectors = np.tile(normal_vectors.transpose(), (X_new.shape[0],1))

            # second block assembly 
            Cross_matrix[:,2*X_obs.shape[0]:] = Cross_K_2[:,::2]*normal_vectors[:,::2] + Cross_K_2[:,1::2]*normal_vectors[:,1::2]

        return Cross_matrix

    def Gram_matrix_assembly(self, obs_structure, observations_dict, **kwargs):

        if obs_structure == 'velocity_and_normal':

            # get positions and size
            X_obs = observations_dict['points']
            X_obs_normal = observations_dict['points_normal']
            Gram_dim = 2*X_obs.shape[0] + X_obs_normal.shape[0]
            Gram_matrix = np.zeros((Gram_dim, Gram_dim))

            # set normal projections
            normal_vectors = observations_dict['normal_vectors']
            if normal_vectors.ndim > 1 :
                normal_vectors = normal_vectors.flatten()[:, np.newaxis]
            normal_vectors_cond = np.tile(normal_vectors.transpose(), (2*X_obs.shape[0],1))
            normal_vectors_normal = np.tile(normal_vectors.transpose(), (2*X_obs_normal.shape[0],1))
            normal_vectors_normal_right = np.tile(normal_vectors.transpose(), (X_obs_normal.shape[0],1))

            # assembly

            Gram_matrix[:2*X_obs.shape[0],:2*X_obs.shape[0]] = self.kernel_function.compute_matrix_incompressible('id','id', X_obs, X_obs, **kwargs)

            Gram_K_12 = self.kernel_function.compute_matrix_incompressible('id','id', X_obs, X_obs_normal, **kwargs)
            Gram_12 = Gram_K_12[:,::2]*normal_vectors_cond[:,::2] + Gram_K_12[:,1::2]*normal_vectors_cond[:,1::2]
            Gram_matrix[:2*X_obs.shape[0],2*X_obs.shape[0]:] = Gram_12
            Gram_matrix[2*X_obs.shape[0]:,:2*X_obs.shape[0]] = Gram_12.transpose()

            Gram_K_22 = self.kernel_function.compute_matrix_incompressible('id','id', X_obs_normal, X_obs_normal, **kwargs)
            Gram_22_right = Gram_K_22[:,::2]*normal_vectors_normal[:,::2] + Gram_K_22[:,1::2]*normal_vectors_normal[:,1::2]
            Gram_22 = Gram_22_right[::2,:]*normal_vectors_normal_right.transpose()[::2,:] + Gram_22_right[1::2,:]*normal_vectors_normal_right.transpose()[1::2,:]
            Gram_matrix[2*X_obs.shape[0]:,2*X_obs.shape[0]:] = Gram_22

        return Gram_matrix

    def regression(self, Cross_matrix, Gram_matrix, values_cond, **kwargs) :

        width = 70
        text = f'regression with Gram size {Gram_matrix.shape[0]}'
        print(f"{'-' * ((width - len(text)))}{text}")

        # compute lower triangular with nugget type

        if self.use_nugget == 'always' :
            Gram_matrix_original = Gram_matrix.copy()
            Gram_matrix = self.add_nugget(Gram_matrix)
            L = sp.linalg.cholesky(Gram_matrix).transpose()
            print(f'[GPR] Permanent nugget {self.nugget} used')

        elif self.use_nugget == 'when_necessary':
            try :
                L = sp.linalg.cholesky(Gram_matrix).transpose()
            except :
                Gram_matrix = self.add_nugget(Gram_matrix)
                L = sp.linalg.cholesky(Gram_matrix).transpose()
                print(f'[GPR] det(Gram) = {np.linalg.det(Gram_matrix)}  / Nugget {self.nugget} used')

        # Cholesky error
        print(f'[GPR] Gram order (max) : {np.abs(Gram_matrix).max()}')
        print(f'[GPR] Cholesky error (max) : {np.abs(np.matmul(L,L.transpose()) - Gram_matrix).max()}')

        Z = sp.linalg.solve_triangular(L, values_cond, lower = True)

        # compute GPR inversion error
        A_check_inversion = sp.linalg.solve_triangular(L, Gram_matrix.transpose(), lower = True) # Gram (may include nugget)
        val_check_inversion = np.matmul(A_check_inversion.transpose(), Z)
        inversion_error = np.abs(val_check_inversion - values_cond)

        # compute inversion error
        Gamblet_coeff = sp.linalg.solve_triangular(L.transpose(), Z, lower = False)
        print(f'[GPR] Gamblet coefficients order : max {np.abs(Gamblet_coeff).max()} -- mean {np.abs(Gamblet_coeff).mean()}')
        val_check_inversion_gamblet = np.matmul(Gram_matrix, Gamblet_coeff) # Gram matrix (observation check)
        inversion_error_gamblet = np.abs(val_check_inversion_gamblet - values_cond)

        # compute regression
        if (inversion_error_gamblet.max() < inversion_error.max()) and (inversion_error_gamblet.mean() < inversion_error.mean()) :
            inversion_error = inversion_error_gamblet
            val = np.matmul(Cross_matrix, Gamblet_coeff) # Gamblet matrix (new points)
            print('[GPR] Gamblet linear combination used !')

        else :
            A = sp.linalg.solve_triangular(L, Cross_matrix.transpose(), lower = True)
            val = np.matmul(A.transpose(), Z)

        print(f'[GPR] Observation order (mean) : {np.abs(values_cond).mean()}')
        print(f'[GPR] GPR inversion error (mean) : {inversion_error.mean()}')
        print(f'[GPR] GPR inversion error (max) : {inversion_error.max()}')

 
        if 'gamblet_coeff_out' in kwargs and kwargs.get('gamblet_coeff_out') :

            return val, Gamblet_coeff

        return val
    
    def add_nugget(self, Gram_matrix) :

        N_gram = Gram_matrix.shape[0]

        if self.nugget_type == 'standard' :
            Gram_matrix_nugget = Gram_matrix + (self.nugget*np.eye(N_gram))
            print(f'[GPR] Standard nugget used')

        return Gram_matrix_nugget

    
    def filter_close_points(self, X_grid, tol_dist) :
        X_grid_tree = sp.spatial.KDTree(X_grid)
        indexes = np.ones(len(X_grid), dtype=bool)

        for i_grid in range(len(X_grid)):
            if not indexes[i_grid]:  
                continue
            neighbors = X_grid_tree.query_ball_point(X_grid[i_grid], tol_dist) # to remove
            for it_neighbors in neighbors:
                if it_neighbors > i_grid:
                    indexes[it_neighbors] = False
        
        return X_grid[indexes], indexes
