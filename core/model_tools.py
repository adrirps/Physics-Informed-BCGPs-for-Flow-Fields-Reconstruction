
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import  matplotlib.ticker as plticker
import matplotlib.tri as ptri
import matplotlib.path as pth
import matplotlib.patches as patches
import time

from .profiles import cylinder, NACA_airfoil


# Utilities for sampling, setting observations, performing estimations and visualization


class main_tools():

    def __init__(self, input_config) :

        self.print_title('GPR Model Initialization')
        self.config = input_config
        return

    def define_domain(self, domain_input) :

        if hasattr(self.config, 'domain') :
            domain_raw = self.config.domain
        else :
            domain_raw = domain_input

        self.print_title('Domain Definition')

        if isinstance(domain_raw, str) and domain_raw == 'from_data' :
            print(f'[GPR Model] Loading domain from data...')
            self.domain = self.loaded_data_boundary.config['domain']
        elif np.issubdtype(domain_raw.dtype, np.number) :
            self.domain = domain_raw
        else :
            raise ValueError('[GPR Model] Domain format is not valid')
        print(f'[GPR Model] Domain defined as [{self.domain[0,0]:.2f},{self.domain[0,1]:.2f}] x [{self.domain[1,0]:.2f},{self.domain[1,1]:.2f}]')


        if hasattr(self, 'internal_data') :
            print('[GPR Model] Filtering loaded data to domain defintion')

            point_centers = self.internal_data.foam_centres
            indexes_domain = np.where((point_centers[:, 0] >= self.domain[0,0]) & (point_centers[:, 0] <= self.domain[0,1]) & 
                    (point_centers[:, 1] >= self.domain[1,0]) & (point_centers[:, 1] <= self.domain[1,1]))[0]

            self.internal_data.foam_centres = self.internal_data.foam_centres[indexes_domain,:]
            self.internal_data.foam_centres_tree = sp.spatial.KDTree(self.internal_data.foam_centres)
            self.internal_data.foam_centres_triang = sp.spatial.Delaunay(self.internal_data.foam_centres)
            self.internal_data.u_data_internal = self.internal_data.u_data_internal[:,indexes_domain,:]

    def points_inside_domain(self, X_grid) :

        indexes_domain = np.where((X_grid[:, 0] >= self.domain[0,0]) & (X_grid[:, 0] <= self.domain[0,1]) & 
                (X_grid[:, 1] >= self.domain[1,0]) & (X_grid[:, 1] <= self.domain[1,1]))[0]
        
        return X_grid[indexes_domain,:], indexes_domain

    def set_domain_interpolation_points(self, N_domain_interpolation, method, **kwargs) :

        self.print_title('Domain Points Setting for Interpolation')
        
        if 'fixed_time' in kwargs :
            fixed_time_it = kwargs.get('fixed_time')

        domain_interpolation = self.domain
        
        if method == 'truth_at_data_points' :

            foam_centres_filtered, indexes_domain = self.filter_domain_obstacle(self.internal_data.foam_centres)

            # get all cell centers inside the domain :
            indexes_left = self.internal_data.foam_centres[:,0] > domain_interpolation[0,0]
            indexes_right = self.internal_data.foam_centres[:,0] < domain_interpolation[0,1]
            indexes_down = self.internal_data.foam_centres[:,1] > domain_interpolation[1,0]
            indexes_up = self.internal_data.foam_centres[:,1] < domain_interpolation[1,1]
            
            indexes = indexes_left*indexes_right*indexes_down*indexes_up*indexes_domain

            X_domain = self.internal_data.foam_centres[indexes,:]
            velocity_domain  = self.internal_data.u_data_internal[:,indexes,:]

            if N_domain_interpolation == 'all' :
                pass
            else :
                # Subsample a of domain interpolation points
                tolerance_init = 0.0005
                X_domain_iter, indexes_iter = self.filter_close_points(X_domain,tolerance_init)
                if X_domain_iter.shape[0] < N_domain_interpolation :
                    raise ValueError('[GPR model] Tolerance for close point filtering is too big!')

                tolerance_iter = tolerance_init
                while (X_domain_iter.shape[0] - N_domain_interpolation)/ N_domain_interpolation > 0.25 :
                    tolerance_iter = tolerance_iter + tolerance_init
                    X_domain_iter, indexes_iter = self.filter_close_points(X_domain,tolerance_iter)

                X_domain = X_domain_iter
                velocity_domain = velocity_domain[:,indexes_iter,:]

            # set truth : X_truth is the same as domain
            if 'fixed_time' in kwargs :
                velocity_truth = velocity_domain[fixed_time_it,:,:]

            self.N_truth = X_domain.shape[0]

            print(f'[GPR Model] Ground truth is set at {self.N_truth} points')

        else :

            X_domain = self.build_grid(N_domain_interpolation, self.domain, with_limits = False)


        if hasattr(self,'X_boundary_walls') and self.X_boundary_walls.shape[0] > 0 :
            X_domain = np.concatenate((self.X_boundary_walls, X_domain))
            velocity_truth = np.concatenate((self.velocity_boundary_walls, velocity_truth))

        self.X_domain = X_domain
        self.velocity_truth = velocity_truth

        X_domain_filtered, indexes_domain = self.filter_domain_obstacle(X_domain)
        self.X_domain_filtered = X_domain_filtered
        self.indexes_domain = indexes_domain
        self.N_domain_filtered = self.X_domain_filtered.shape[0]

        if 'airfoil_box_distance_interpolation' in kwargs :
            self.set_airfoil_box_evaluation(distance_tol = kwargs.get('airfoil_box_distance_interpolation'), fixed_time = fixed_time_it)

        self.N_domain = self.X_domain.shape[0]

        print(f'[Model] Domains points computed at {self.N_domain} points')

    def filter_domain_obstacle(self, X_domain, **kwargs):
        
        if self.config.obstacle_type == 'NACA_airfoil':

            obstacle_path = pth.Path(self.X_obstacle)
            inner_indexes = obstacle_path.contains_points(X_domain)
            indexes = ~ inner_indexes

            X_domain = X_domain[indexes,:]
                    
        elif self.config.obstacle_type == 'cylinder' :

            gamma_x0 = self.config.obstacle_parameters[0] # center
            gamma_y0 = self.config.obstacle_parameters[1]
            gamma_radius = self.config.obstacle_parameters[2]

            points_distance = (X_domain[:,0] - gamma_x0)**2 + (X_domain[:,1] - gamma_y0)**2
            X_domain = X_domain[points_distance > (gamma_radius)**2 ,:]
            indexes = points_distance > (gamma_radius)**2
            inner_indexes = points_distance <= (gamma_radius)**2

        print('[Points] Domain points filtered according to obstacle boundary constraint')

        if 'inner' in kwargs and kwargs.get('inner') :
            return X_domain, indexes, inner_indexes
        else : 
            return X_domain, indexes
        

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

    def set_boundary(self, **kwargs) :

        self.print_title('Boundary Points Setting')

        fixed_time = kwargs.get('fixed_time')


        X_boundary = []
        velocity_boundary = []
        X_boundary_walls = []
        velocity_boundary_walls = []
        N_time = len(self.internal_data.time_series)

        print('[GPR Model] Boundary set according to time series from internal data loading')

        for key in self.config.boundary_definition :
            if key == 'top' or key == 'bottom' :
                
                if self.config.boundary_definition[key] == 'no-slip' :
                    
                    N_boundary_section = kwargs.get('N_boundary_section')

                    # set boundary points
                    temp_X_boundary_section = np.zeros((N_boundary_section, 2))
                    temp_X_boundary_section[:,0] = np.linspace(self.domain[0,0], self.domain[0,1], N_boundary_section)
                    if key == 'top' :
                        temp_X_boundary_section[:,1] = self.domain[1,1]
                    elif key == 'bottom' :
                        temp_X_boundary_section[:,1] = self.domain[1,0]

                    # set velocity (no-slip)
                    temp_velocity_boundary_section = np.zeros((N_time, N_boundary_section, 2))
                
                elif self.config.boundary_definition[key] == 'all_data_points' :

                    temp_X_boundary_section, temp_velocity_boundary_section = self.internal_data.compute_boundary_section(boundary_type = key, method = 'all_data_points')

                X_boundary_walls.append(temp_X_boundary_section)
                velocity_boundary_walls.append(temp_velocity_boundary_section)
                    
            elif key == 'inlet' or key == 'outlet' :

                if self.config.boundary_definition[key] == 'all_data_points' :

                    temp_X_boundary_section, temp_velocity_boundary_section = self.internal_data.compute_boundary_section(boundary_type = key, method = 'all_data_points')
                
                    if key == 'inlet' :
                        if len(temp_velocity_boundary_section.shape) == 3 :
                            norm_velocity = np.linalg.norm(temp_velocity_boundary_section, axis = 2)
                            self.mean_velocity_inlet_data = norm_velocity.mean(axis = 1) # for Reynolds number


            X_boundary.append(temp_X_boundary_section)
            velocity_boundary.append(temp_velocity_boundary_section)


        X_boundary = np.vstack(X_boundary)
        velocity_boundary = np.concatenate(velocity_boundary, axis = 1)
        X_boundary_walls = np.vstack(X_boundary_walls)
        velocity_boundary_walls = np.concatenate(velocity_boundary_walls, axis = 1)

        # set fixed time
        if fixed_time is not None :
            X_boundary, velocity_boundary = self.set_fixed_time(fixed_time, X_boundary, velocity_boundary)
            X_boundary_walls, velocity_boundary_walls = self.set_fixed_time(fixed_time, X_boundary_walls, velocity_boundary_walls)

        self.X_boundary = X_boundary
        self.velocity_boundary = velocity_boundary
        self.X_boundary_walls = X_boundary_walls
        self.velocity_boundary_walls = velocity_boundary_walls
        self.N_boundary = self.X_boundary.shape[0]

        self.reset_domain_from_boundary()

        print(f'[Model] Boundary points set from user defined method at {self.N_boundary} points')


    def compute_Reynolds_number(self, viscosity) :

        nu = viscosity
        if self.config.obstacle_type == 'cylinder' :
            L = 2 * self.config.obstacle_parameters[2] # diamater
        elif self.config.obstacle_type == 'NACA_airfoil' :
            L = self.config.obstacle_parameters[3] # chord length
        mean_u_inlet_time = self.mean_velocity_inlet_data

        Reynolds = mean_u_inlet_time * L / nu

        print(f'[Data] Reynolds number of simulation : {Reynolds.mean():.2f} ')

        return Reynolds

    def set_fixed_time(self, fixed_time, X_grid, velocity_grid) :

        if isinstance(fixed_time, (int, float)) and len(X_grid.shape) == 3 :
            X_grid = X_grid[fixed_time,:,:]
            print(f'[GPR Model] Points restricted to time iteration = {fixed_time}')
        if isinstance(fixed_time, (int, float)) and len(velocity_grid.shape) == 3 :
            velocity_grid = velocity_grid[fixed_time,:,:]
            print(f'[GPR Model] Velocity restricted to time iteration = {fixed_time}')
        return X_grid, velocity_grid

    def reset_domain_from_boundary(self) :
        new_domain = np.zeros((2,2))
        new_domain[0,0] = self.X_boundary[:,0].min()
        new_domain[0,1] = self.X_boundary[:,0].max()
        new_domain[1,0] = self.X_boundary[:,1].min()
        new_domain[1,1] = self.X_boundary[:,1].max()
        
        self.domain = new_domain
        print(f'[GPR Model] Domain redefined as [{self.domain[0,0]:.2f},{self.domain[0,1]:.2f}] x [{self.domain[1,0]:.2f},{self.domain[1,1]:.2f}]')

    def set_obstacle_points(self, N_obstacle_points) :

        self.print_title('Obstacle points')
        # set interpolation points : over obstacle boundary
        N_grid = N_obstacle_points

        
        if self.config.obstacle_type == 'cylinder' :

            s_grid = np.linspace(0, 2*np.pi, N_grid + 1)[:-1]
            X_obstacle = np.zeros((N_grid,2))

            self.cylinder = cylinder(self.config.obstacle_parameters)
            X_obstacle[:,0] = self.cylinder.gamma_1_np(s_grid)
            X_obstacle[:,1] = self.cylinder.gamma_2_np(s_grid)

            print('[GPR Model] Obstacle points set according to kernel gamma transformation')

        elif self.config.obstacle_type == 'NACA_airfoil' :

            obstacle_parameters = self.config.obstacle_parameters
            center_0 = obstacle_parameters[0]
            center_1 = obstacle_parameters[1]
            NACA_code = obstacle_parameters[2]
            chord_length = obstacle_parameters[3]
            self.airfoil = NACA_airfoil(NACA_code)

            X_obstacle, s_grid = self.airfoil.boundary_values(N_grid,chord_length,limits = [0,2*np.pi] ) # complete domain set for evaluation
            if s_grid[-1] == 2*np.pi :
                X_obstacle = X_obstacle[:-1,:] # last point not included since compact_set domain is periodic
                s_grid = s_grid[:-1]
            X_obstacle[:,0] = X_obstacle[:,0] + center_0
            X_obstacle[:,1] = X_obstacle[:,1] + center_1

        # set inside domain only

        indexes_domain = np.where((X_obstacle[:, 0] >= self.domain[0,0]) & (X_obstacle[:, 0] <= self.domain[0,1]) & 
                        (X_obstacle[:, 1] >= self.domain[1,0]) & (X_obstacle[:, 1] <= self.domain[1,1]))[0]
        
        X_obstacle = X_obstacle[indexes_domain,:]
        s_grid = s_grid[indexes_domain]

        self.X_obstacle = X_obstacle
        self.N_obstacle = self.X_obstacle.shape[0]
        self.obstacle_s_grid = s_grid # for error computation

        print(f'[GPR Model] Obstacle points set at {self.N_obstacle} points')

    def compute_airfoil_box(self, distance_tol, **kwargs) :

        # define box

        curve_values = self.X_obstacle

        curve_values_x_min = curve_values[:,0].min()
        curve_values_x_max = curve_values[:,0].max()
        curve_values_y_min = curve_values[:,1].min()
        curve_values_y_max = curve_values[:,1].max()

        box_domain = np.zeros((2,2))
        box_domain[0,0] = curve_values_x_min - distance_tol
        box_domain[0,1] = curve_values_x_max + distance_tol
        box_domain[1,0] = curve_values_y_min - distance_tol
        box_domain[1,1] = curve_values_y_max + distance_tol

        x_min, x_max = box_domain[0,:]
        y_min, y_max = box_domain[1,:]

        box_vertices = np.array([
            [x_min, y_min],  # Bottom-left
            [x_max, y_min],  # Bottom-right
            [x_max, y_max],  # Top-right
            [x_min, y_max],  # Top-left
        ])

        # filter data in rectangle domain

        closed_box_path = np.vstack((box_vertices,box_vertices[0,:]))
        box_path = pth.Path(closed_box_path)
        indexes = box_path.contains_points(self.internal_data.foam_centres)

        X_airfoil_box = self.internal_data.foam_centres[indexes,:]
        velocity_airfoil_box = self.internal_data.u_data_internal[:,indexes,:]

        # filter data in airfoil domain

        obstacle_tree = sp.spatial.cKDTree(self.X_obstacle)

        filter_distances, temp = obstacle_tree.query(X_airfoil_box)
        indexes_filter = filter_distances <= distance_tol

        X_airfoil_box = X_airfoil_box[indexes_filter]
        velocity_airfoil_box = velocity_airfoil_box[:,indexes_filter,:]

        print(f'[GPR Model] Airfoil box computed at distance {distance_tol} from airfoil')

        # set fixed point
        if 'fixed_time' in kwargs :
            fixed_time = kwargs.get('fixed_time')
            X_airfoil_box, velocity_airfoil_box = self.set_fixed_time(fixed_time, X_airfoil_box, velocity_airfoil_box)
            
            print(f'[GPR Model] Airfoil box set at fixed time iteration = {fixed_time}')

        return X_airfoil_box, velocity_airfoil_box
    
    def filter_airfoil_box(self, X_grid, distance_tol) :

        obstacle_tree = sp.spatial.cKDTree(self.X_obstacle)

        filter_distances, temp = obstacle_tree.query(X_grid)
        indexes_filter = filter_distances > distance_tol

        X_grid_filter = X_grid[indexes_filter]

        return X_grid_filter, indexes_filter

    def set_airfoil_box_evaluation(self, distance_tol, **kwargs) :

        self.print_title('Airfoil box - evaluation')
        X_airfoil_box_eval, velocity_airfoil_box_eval = self.compute_airfoil_box(distance_tol, **kwargs)
        X_airfoil_box_eval_filtered, indexes_filter = self.filter_close_points(X_airfoil_box_eval, 0.002)
        velocity_airfoil_box_eval_filtered = velocity_airfoil_box_eval[indexes_filter,:]

        new_X_domain = np.vstack((X_airfoil_box_eval_filtered, self.X_domain))
        new_velocity_truth = np.vstack((velocity_airfoil_box_eval_filtered, self.velocity_truth))
        
        self.X_domain = new_X_domain
        self.velocity_truth = new_velocity_truth
        self.N_truth = self.velocity_truth.shape[0]

        print('[GPR Model] Reset domain interpolation points with airfoil box') # ???

    def set_airfoil_box(self, distance_tol, **kwargs) :

        self.print_title('Airfoil box - observation')
        
        X_airfoil_box, velocity_airfoil_box = self.compute_airfoil_box(distance_tol, **kwargs)
        self.X_airfoil_box = X_airfoil_box
        self.velocity_airfoil_box = velocity_airfoil_box
        print(['[GPR Model] Airfoil box set for observations'])



    def perform_GPR(self, GP, observations_dict, **kwargs) :

        self.print_title('Computing GPR')

        # Mean computation

        start_time_mean = time.time()

        velocity_interpolation_domain = GP.interpolation('matrix_K_mean', self.X_domain, observations_dict)
        velocity_interpolation_obstacle = GP.interpolation('matrix_K_mean', self.X_obstacle, observations_dict)

        scalar_stream_obstacle = GP.interpolation('scalar_stream_mean', self.X_obstacle, observations_dict)

        end_time_mean = time.time()

        velocity_interpolation_domain = velocity_interpolation_domain.reshape((self.N_domain,2))
        velocity_interpolation_obstacle = velocity_interpolation_obstacle.reshape((self.N_obstacle,2))

        print(f"Execution mean interpolation time: {end_time_mean - start_time_mean:.6f} seconds")

        # Uncertainty computation

        if 'out_list' in kwargs :
            out_input = kwargs.get('out_list')
            if 'UQ' in out_input :

                start_time_cov = time.time()

                UQ_velocity_interpolation_domain = GP.interpolation('matrix_K_covariance', self.X_domain, observations_dict)
                UQ_velocity_interpolation_domain_trace = np.sqrt( np.diag(UQ_velocity_interpolation_domain[::2, ::2] + UQ_velocity_interpolation_domain[1::2, 1::2]) )

                end_time_cov = time.time()

                print(f"Execution covariance interpolation time: {end_time_cov - start_time_cov:.6f} seconds")

        # Plots

        if hasattr(self.config, 'visualize') and self.config.visualize :
            
            plot_dict = {
                'velocity_interpolation_obstacle'           :   velocity_interpolation_obstacle,
                'velocity_interpolation_domain'             :   velocity_interpolation_domain
            }

            if 'plot_list' in kwargs :
                plot_list = kwargs.get('plot_list')
                if 'total_SD' in plot_list :
                    plot_dict['UQ_velocity_interpolation_domain_trace'] = UQ_velocity_interpolation_domain_trace
                    self.do_plot(plot_dict, method = 'total_SD')
                elif 'velocity' in plot_list :
                    self.do_plot(plot_dict, method = 'interpolation')


            if hasattr(self, 'velocity_truth'):

                # compute relative error
                velocity_truth_norms = np.linalg.norm(self.velocity_truth, axis = 1)

                if np.any(velocity_truth_norms == 0) :
                    idx = velocity_truth_norms == 0
                    small_element_limit = velocity_truth_norms[ ~ idx].min()
                    velocity_truth_norms[idx] += small_element_limit

                error_domain_abs = (velocity_interpolation_domain - self.velocity_truth)
                error_domain_rel = np.zeros((self.velocity_truth.shape[0],2))
                error_domain_rel[:,0] = error_domain_abs[:,0] / velocity_truth_norms
                error_domain_rel[:,1] = error_domain_abs[:,1] / velocity_truth_norms

                # plot error
                plot_dict_error = {
                    'velocity_interpolation_obstacle'   : velocity_interpolation_obstacle,
                    'error_domain_relative'             : error_domain_rel
                }
                self.do_plot(plot_dict_error, method = 'error_relative')
        
        # out object
        if 'out_list' in kwargs :
            out_input = kwargs.get('out_list')
            
            out_dict = {}
            if 'domain' in out_input :
                out_dict['domain'] = velocity_interpolation_domain
            if 'obstacle' in out_input :
                out_dict['obstacle'] = velocity_interpolation_obstacle
            if 'stream' in out_input :
                out_dict['stream'] = scalar_stream_obstacle
            if 'UQ' in out_input :
                out_dict['UQ_domain_trace'] = UQ_velocity_interpolation_domain_trace
            
            return out_dict


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
    
    def set_local_region(self, method, N_region, region_parameters) :

        if method == 'box' :
            indexes_domain = np.where((self.X_domain[:, 0] >= region_parameters[0,0]) & (self.X_domain[:, 0] <= region_parameters[0,1]) & 
                            (self.X_domain[:, 1] >= region_parameters[1,0]) & (self.X_domain[:, 1] <= region_parameters[1,1]))[0]

        elif method == 'circle' :
            x0 = region_parameters[0]
            y0 = region_parameters[1]
            radius = region_parameters[2]
            indexes_domain = np.where((self.X_domain[:, 0] - x0)**2 + (self.X_domain[:,1] - y0)**2 <= radius**2 )[0]

        X_grid = self.X_domain[indexes_domain,:]
        velocity_grid = self.velocity_truth[indexes_domain,:]

        tol_distance = 0.0001
        while X_grid.shape[0] > N_region :
            X_grid, indexes = self.filter_close_points(X_grid, tol_distance)
            velocity_grid = velocity_grid[indexes,:]
            tol_distance += 0.00005

        tol_distance -= 0.00005

        print(f'[GPR model] Local region ({method}) set at {X_grid.shape[0]} points')

        return X_grid, velocity_grid, tol_distance/2

    def add_discrete_obstacle_observation(self, method, N_discrete) :

        if self.config.obstacle_type == 'cylinder' :
            s_grid = np.linspace(0, 2*np.pi, N_discrete + 1)[:-1] # periodic set
            X_obstacle_discrete = np.zeros((N_discrete,2))
            X_obstacle_discrete[:,0] = self.cylinder.gamma_1_np(s_grid)
            X_obstacle_discrete[:,1] = self.cylinder.gamma_2_np(s_grid)

            X_obstacle_discrete_normal = X_obstacle_discrete

            # normal
            normal_grid_x = np.cos(s_grid) # unit normal
            normal_grid_y = np.sin(s_grid)

        elif self.config.obstacle_type == 'NACA_airfoil' :
            chord_length = self.config.obstacle_parameters[3]
            X_obstacle_discrete, temp = self.airfoil.boundary_values(N_discrete + 1, chord_length,limits = [0,2*np.pi] ) # complete curve

            # normal
            tangent_grid, filter_indexes = self.airfoil.curve_derivatives_values(N_discrete + 1, chord_length, normalized = True)
            X_obstacle_discrete_normal = X_obstacle_discrete[filter_indexes,:] # filter indexes according to non-zero and non-nan derivatives
            
            normal_grid_x = tangent_grid[:,1]
            normal_grid_y = - tangent_grid[:,0] # outward unit normal
        
        normal_grid = np.vstack((normal_grid_x, normal_grid_y)).transpose()

        if method == 'discrete_normal':

            X_obstacle_discrete = X_obstacle_discrete_normal

            indexes_domain = np.where((X_obstacle_discrete[:, 0] >= self.domain[0,0]) & (X_obstacle_discrete[:, 0] <= self.domain[0,1]) & 
                (X_obstacle_discrete[:, 1] >= self.domain[1,0]) & (X_obstacle_discrete[:, 1] <= self.domain[1,1]))[0]
            
            X_obstacle_discrete = X_obstacle_discrete[indexes_domain,:]
            normal_grid = normal_grid[indexes_domain,:]
            
            # filter close points
            inter_distance = 0.00001
            X_obstacle_discrete, indexes = self.filter_close_points(X_obstacle_discrete, inter_distance)
            normal_grid = normal_grid[indexes,:]

            return X_obstacle_discrete, normal_grid


    def compute_obstacle_normal_error(self, velocity_obstacle, **kwargs) :

        # compute normal unit vectors
        s_grid = self.obstacle_s_grid
        
        if self.config.obstacle_type == 'cylinder' :
            normal_grid_x = np.cos(s_grid) # unit normal
            normal_grid_y = np.sin(s_grid)

            tangent_grid = np.vstack((- normal_grid_y, normal_grid_x)).transpose()

        elif self.config.obstacle_type == 'NACA_airfoil' :

            c_length = self.config.obstacle_parameters[3]
            tangent_grid, filter_indexes = self.airfoil.curve_derivatives_values(1, c_length, s_series = s_grid, normalized = True)
            velocity_obstacle = velocity_obstacle[filter_indexes,:]
            normal_grid_x = tangent_grid[:,1]
            normal_grid_y = - tangent_grid[:,0] # outward unit normal

        normal_grid = np.vstack((normal_grid_x, normal_grid_y)).transpose()

        # compute normal and tangent projection
        normal_projections = np.sum(normal_grid*velocity_obstacle, axis = 1)
        tangent_projections = np.sum(tangent_grid*velocity_obstacle, axis = 1)

        if 'relative' in kwargs and kwargs.get('relative') :

            # Relative aggregated error

            agg_norm = np.sum(np.linalg.norm(velocity_obstacle, axis = 1))
            relative_normal_indicator = np.sum(np.abs(normal_projections)) / agg_norm
            relative_tangent_indicator = np.sum(np.abs(tangent_projections)) / agg_norm

            return relative_normal_indicator, relative_tangent_indicator

        return np.abs(normal_projections), np.abs(tangent_projections)



    def do_plot(self, plot_dict, method, **kwargs) :

        if method == 'error_relative' :
            error_domain = plot_dict['error_domain_relative']
        elif method == 'data':
            velocity_data_domain = plot_dict['velocity_data_domain']
        elif method == 'interpolation' :
            velocity_interpolation_domain = plot_dict['velocity_interpolation_domain']
        elif method == 'total_SD' :
            UQ_velocity_interpolation_domain_trace = plot_dict['UQ_velocity_interpolation_domain_trace']

        plot_colorbar = True
        plot_size = 10
        plot_size_scale = (plot_size * (self.domain[1,1]-self.domain[1,0])/(self.domain[0,1]-self.domain[0,0]))*1.1
        if plot_colorbar :
            fig = plt.figure(figsize = (plot_size+2,plot_size_scale))
        else:
            fig = plt.figure(figsize = (plot_size,plot_size_scale))
        
        if method != 'data' :

            point_size = 50

            plt.scatter(self.X_obs[:, 0], self.X_obs[:, 1], marker="o", zorder = 15, alpha = 0.3, facecolors='none', edgecolors='black', s = point_size, label="Grid Points")

            if hasattr(self,'X_normal_obs'):
                plt.scatter(self.X_normal_obs[:, 0], self.X_normal_obs[:, 1], marker="*", zorder = 10, alpha = 0.6, facecolors='none', edgecolors='black', s = point_size + 20)

        X_grid_for_tri = self.X_domain
        X_triang = ptri.Triangulation(X_grid_for_tri[:,0], X_grid_for_tri[:,1])

        if method == 'error_relative' :
            error_norm = np.linalg.norm(error_domain, axis = 1)
            error_max_plot = 0.5

            levels = np.linspace(0, error_max_plot, 100)
            colormap_plot = plt.tricontourf(X_triang, error_norm, levels = levels, cmap = 'brg',
                                        vmin = 0, vmax = error_max_plot )
            
        elif method == 'total_SD' :
            vmax_plot = UQ_velocity_interpolation_domain_trace.max()*1.1
            levels = np.linspace(0, vmax_plot, 100)
            colormap_plot = plt.tricontourf(X_triang, UQ_velocity_interpolation_domain_trace, alpha = 0.5, cmap = 'viridis',
                                            vmin = 0, vmax = vmax_plot, levels = levels )

        else :
            if method == 'interpolation' :
                velocity_plot_domain = velocity_interpolation_domain
            elif method == 'data' :
                velocity_plot_domain = velocity_data_domain

            # colormap
            norm_velocity_domain = np.linalg.norm(velocity_plot_domain, axis = 1)
            if method == 'data' :
                vmax_plot = norm_velocity_domain.max()*1.1
                self.do_plot_vmax = vmax_plot
            elif method == 'interpolation' :
                try : 
                    vmax_plot = self.do_plot_vmax
                except :
                    vmax_plot = norm_velocity_domain.max()*1.1

            levels = np.linspace(0, vmax_plot, 100)
            colormap_plot = plt.tricontourf(X_triang, norm_velocity_domain, levels = levels, alpha = 0.6, cmap = 'rainbow',
                                        vmin = 0, vmax = vmax_plot, antialiased=False    )

        
        X_obstacle = self.X_obstacle
        object = plt.fill(X_obstacle[:,0],X_obstacle[:,1], 0.8, color='gray')

        # format
        plt.axis('equal')
        plt.xlim(self.domain[0,0], self.domain[0,1])
        plt.ylim(self.domain[1,0], self.domain[1,1])

        fontsize = 20

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        if plot_colorbar :
            cb_ax = fig.add_axes([0.91, 0.124, 0.02, 0.754])
            cbar = plt.colorbar(colormap_plot, cax = cb_ax, format=plticker.FormatStrFormatter('%.3f'))
            cbar.ax.tick_params(labelsize = fontsize)

        # save plot
        if method == 'error_relative' :
            save_name = 'relative_error.pdf'
        elif method == 'interpolation' :
            save_name = 'velocity_interpolation.pdf'
            print(f'[Observation] Design direct observations size : {self.X_obs.shape[0]}')
            if hasattr(self, 'X_normal_obs') :
                print(f'[Observation] Design discrete BC observations size {self.X_normal_obs.shape[0]}' )
        elif method == 'data' :
            save_name = 'data.pdf'
        elif method == 'total_SD' :
            save_name = 'sqrt_trace_variance.pdf'
        elif method == 'variance' :
            save_name = 'variance_field.pdf'
        try :
            plt.savefig(r"simulation\BCGP_figures\\" + save_name, bbox_inches='tight', pad_inches=0.01)
        except :
            raise ValueError('[Plot] Set correct path for saving figures')

    def plot_profile_indicators(self, spectral_precision_grid, rel_agg_obstacle_normal, abs_scalar_stream ) :

        plt.figure()
        plt.semilogy(spectral_precision_grid, rel_agg_obstacle_normal, marker='o', linestyle='-', c = 'black', markerfacecolor='white', markeredgecolor='black', label = 'Normal indicator')
        plt.semilogy(spectral_precision_grid, np.array(abs_scalar_stream).mean(axis = 1), '*-', c='black', label = 'Stream indicator')            
        plt.xlabel(r"$-$ Log of spectral expansion precision")
        plt.ylabel("Evaluation of profile indicators", rotation = 90)
        plt.grid(True, which="minor", alpha = 0.6, ls = '--')
        plt.legend(loc='upper right')

        return


    def points_close_to_grid(self, N_grid, X_domain) :

        # grid for visualization
        X_grid = self.build_grid(N_grid, self.domain, with_limits = False)
        X_grid, temp = self.filter_domain_obstacle(X_grid)

        # search for closest points
        domain_tree = sp.spatial.KDTree(X_domain)
        distances, nearest_indexes = domain_tree.query(X_grid)
        X_close = X_domain[nearest_indexes,:]

        return X_close, nearest_indexes

    def plot_solution_snapshot(self) :

        plot_dict = { 'velocity_data_domain'  :  self.velocity_truth }

        self.do_plot(plot_dict, method = 'data') # use same X_domain as interpolation for comparison

    def print_title(self, text) :

        width = 100
        print(f"{'-' * ((width - len(text)))}{text}")




