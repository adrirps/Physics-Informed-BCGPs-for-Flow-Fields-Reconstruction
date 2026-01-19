
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import  matplotlib.ticker as plticker
import matplotlib.tri as ptri
import matplotlib.path as pth

from .profiles import cylinder, NACA_airfoil


# Utilities for sampling, setting observations, performing estimations and visualization

class main_tools():

    def __init__(self, input_config) :

        self.print_title('Model tools initialization')
        self.config = input_config
        return

    def define_domain(self, domain_input) :

        self.print_title('Domain Definition')

        if hasattr(self.config, 'domain') :
            domain_raw = self.config.domain
        else :
            domain_raw = domain_input

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

            self.foam_centres = self.internal_data.foam_centres[indexes_domain,:]
            self.foam_centres_tree = sp.spatial.KDTree(self.internal_data.foam_centres)
            self.foam_centres_triang = sp.spatial.Delaunay(self.internal_data.foam_centres)
            self.u_data_internal = self.internal_data.u_data_internal[:,indexes_domain,:]

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

            foam_centres_filtered, indexes_domain = self.filter_domain_obstacle(self.foam_centres)

            # get all cell centers inside the domain :
            indexes_left = self.foam_centres[:,0] > domain_interpolation[0,0]
            indexes_right = self.foam_centres[:,0] < domain_interpolation[0,1]
            indexes_down = self.foam_centres[:,1] > domain_interpolation[1,0]
            indexes_up = self.foam_centres[:,1] < domain_interpolation[1,1]
            
            indexes = indexes_left*indexes_right*indexes_down*indexes_up*indexes_domain

            X_domain = self.foam_centres[indexes,:]
            velocity_domain  = self.u_data_internal[:,indexes,:]

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
            else:
                velocity_truth = velocity_domain

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
            self.set_airfoil_box_evaluation(distance_tol = kwargs.get('airfoil_box_distance_interpolation'), **kwargs)

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

    def check_extrapolation(self, X_grid, X_obs) :

        if self.config.obstacle_type == 'NACA_airfoil' :

            idx_half = X_obs[:,1] >= (self.domain[1,0] + self.domain[1,1])/2
            X_up = X_obs[idx_half,:]
            X_down = X_obs[ ~ idx_half,:]

            temp, idx_up = self.filter_convex_hull(X_grid, X_up)
            temp, idx_down = self.filter_convex_hull(X_grid, X_down)
            idx =  idx_up | idx_down

        else :
            return X_grid, np.array(range(X_grid.shape[0]))

        return X_grid[idx], idx

    def filter_convex_hull(self, X_grid, X_filter) :
        # This utility checks whether X_grid is in convex hull of X_filter

        outer_hull = sp.spatial.ConvexHull(X_filter)
        hull_points = X_filter[outer_hull.vertices]
        hull_path = pth.Path(np.vstack([hull_points, hull_points[0]]))
        idx = hull_path.contains_points(X_grid)
        X_inside = X_grid[idx]

        return X_inside, idx 

    def filter_close_points(self, X_grid, tol_dist) :
        X_grid_tree = sp.spatial.KDTree(X_grid)
        indexes = np.ones(len(X_grid), dtype=bool)

        for i_grid in range(len(X_grid)):
            if not indexes[i_grid]:  
                continue
            neighbors = X_grid_tree.query_ball_point(X_grid[i_grid], tol_dist) # idx to remove
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


        # set inside domain only and convex hull

        indexes_domain = np.where((X_obstacle[:, 0] >= self.domain[0,0]) & (X_obstacle[:, 0] <= self.domain[0,1]) & 
                        (X_obstacle[:, 1] >= self.domain[1,0]) & (X_obstacle[:, 1] <= self.domain[1,1]))[0]
        
        X_obstacle = X_obstacle[indexes_domain,:]
        s_grid = s_grid[indexes_domain]

        # for visualization
        self.X_obstacle_visu = X_obstacle.copy()

        # check convex hull
        X_obstacle, idx = self.check_extrapolation(X_obstacle, self.foam_centres)
        s_grid = s_grid[idx]

        # compute unit normal vectors (and tangents)

        if self.config.obstacle_type == 'cylinder' :

            normal_grid_x = np.cos(s_grid) # outward unit normal
            normal_grid_y = np.sin(s_grid)
            tangent_grid = np.vstack((- normal_grid_y, normal_grid_x)).transpose()

        elif self.config.obstacle_type == 'NACA_airfoil' :

            # normal and tanget vectors 
            tangent_grid, filter_indexes_temp = self.airfoil.curve_derivatives_values(1, chord_length, s_series = s_grid, normalized = True)
            normal_grid_x = tangent_grid[:,1]
            normal_grid_y = - tangent_grid[:,0] # outward unit normal


        self.X_obstacle = X_obstacle
        self.gamma_grid = s_grid
        self.N_obstacle = self.X_obstacle.shape[0]

        self.tangent_grid = tangent_grid
        self.normal_grid = np.vstack((normal_grid_x, normal_grid_y)).transpose()

        print(f'[GPR Model] Obstacle points (and unit normal vectors) set at {self.N_obstacle} points')


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
        indexes = box_path.contains_points(self.foam_centres)

        X_airfoil_box = self.foam_centres[indexes,:]
        velocity_airfoil_box = self.u_data_internal[:,indexes,:]

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
        new_X_domain = np.vstack((X_airfoil_box_eval_filtered, self.X_domain))
        if 'fixed_time' in kwargs :
            velocity_airfoil_box_eval_filtered = velocity_airfoil_box_eval[indexes_filter,:]
            new_velocity_truth = np.vstack((velocity_airfoil_box_eval_filtered, self.velocity_truth))
        else :
            velocity_airfoil_box_eval_filtered = velocity_airfoil_box_eval[:,indexes_filter,:]
            new_velocity_truth = np.concatenate((velocity_airfoil_box_eval_filtered, self.velocity_truth), axis = 1)

        self.X_domain = new_X_domain
        self.velocity_truth = new_velocity_truth
        self.N_truth = self.X_domain.shape[0]

        print('[GPR Model] Domain interpolation updated with airfoil box')


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

        return X_grid, tol_distance/2


    def build_noised_grid_domain(self, seed, N_grid, **kwargs) :

        X_grid_raw = self.build_grid(N_grid, self.domain, with_limits = False)

        # noise
        np.random.seed(seed)
        if 'noise_level' in kwargs:
            noise_level = kwargs.get('noise_level')
        else :
            noise_level = 0.005
        X_grid = X_grid_raw + np.random.normal(0, noise_level, X_grid_raw.shape[0]*2).reshape(X_grid_raw.shape)

        # add local region
        if 'N_local_region' in kwargs :
            N_local_region = kwargs.get('N_local_region')
            X_local_region, tol_temp = self.set_local_region('circle', N_region = N_local_region,
                                                                    region_parameters = [-0.007, 0, 0.03] )
            X_grid = np.concatenate((X_local_region, X_grid))

        # filter out of domain, obstacle and close points
        X_grid, temp = self.points_inside_domain(X_grid)
        if 'left_x_limit' in kwargs :
            left_x_limit = kwargs.get('left_x_limit')
            X_grid = X_grid[ X_grid[:,0] > left_x_limit ,:]

        X_grid, temp = self.filter_domain_obstacle(X_grid)
        X_grid, temp = self.check_extrapolation(X_grid, self.foam_centres)
        X_grid, temp = self.filter_close_points(X_grid, tol_dist = 0.003)

        print(f'[Model] Domain grid set at {X_grid.shape[0]} positions')

        return X_grid



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


    def get_true_fields(self, it_time, X_grid, out_list) :

        # get triangular convex hull
        indexes = self.foam_centres_triang.find_simplex(X_grid)
        if np.any(indexes < 0) :

            raise ValueError('[Error] Points outside truth domain')

        nearest_cell_centres_indexes = self.foam_centres_triang.simplices[indexes]
        nearest_cell_centres = self.foam_centres[nearest_cell_centres_indexes]
        nearest_cell_velocities = self.u_data_internal[it_time,:,:][nearest_cell_centres_indexes]
        out_dict = {}

        # compute fields
        if 'velocity' in out_list :
            velocity_grid = np.zeros((X_grid.shape[0],2))
            for it_x in range(X_grid.shape[0]) :
                if np.any(nearest_cell_centres[it_x,:,:] - X_grid[it_x,:] == 0) :
                    idx = (nearest_cell_centres[it_x,:,:] - X_grid[it_x,:] == 0).all(axis = 1)
                    velocity_grid[it_x,:] = nearest_cell_velocities[it_x,idx,:]
                else :
                    velocity_grid[it_x,0] = sp.interpolate.griddata(nearest_cell_centres[it_x,:,:], nearest_cell_velocities[it_x,:,0], X_grid[it_x,:], method = 'linear')
                    velocity_grid[it_x,1] = sp.interpolate.griddata(nearest_cell_centres[it_x,:,:], nearest_cell_velocities[it_x,:,1], X_grid[it_x,:], method = 'linear')
            out_dict['velocity'] = velocity_grid

        return out_dict


    def compute_obstacle_normal_error(self, velocity_obstacle, **kwargs) :

        # compute normal and tangent projection
        normal_projections = np.sum(self.normal_grid*velocity_obstacle, axis = 1)
        tangent_projections = np.sum(self.tangent_grid*velocity_obstacle, axis = 1)

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
        if method == 'source_error' :
            source_error_domain = plot_dict['source_error_domain']
        elif method == 'data':
            velocity_data_domain = plot_dict['velocity_data_domain']
        elif method == 'interpolation' :
            velocity_interpolation_domain = plot_dict['velocity_interpolation_domain']
        elif method == 'u_error' :
            field_error = plot_dict['u_error']
        elif method == 'total_SD' :
            UQ_velocity_interpolation_domain_trace = plot_dict['UQ_velocity_interpolation_domain_trace']
        elif method == 'scalar_variance' :
            velocity_scalar_variance = plot_dict['velocity_scalar_variance']
        elif method == 'UQ_field' :
            field_UQ = plot_dict['UQ_field']


        plot_size = 10
        plot_size_scale = (plot_size * (self.domain[1,1]-self.domain[1,0])/(self.domain[0,1]-self.domain[0,0]))*1.1
        fig = plt.figure(figsize = (plot_size,plot_size_scale))
        
        if not method.startswith('data') :

            point_size = 50

            if hasattr(self, 'X_obs') :
                if hasattr(self, 'U_obs') :
                    plt.quiver(self.X_obs[:, 0], self.X_obs[:, 1], self.U_obs[:,0], self.U_obs[:,1], units='dots', width = 1.25, scale = 7, scale_units = 'x', zorder=15)
                else :
                    plt.scatter(self.X_obs[:, 0], self.X_obs[:, 1], marker=".", color = 'black', zorder = 15, alpha = 0.8, s = point_size/7, label="U Observation")

            if hasattr(self,'X_normal_obs'):
                plt.scatter(self.X_normal_obs[:, 0], self.X_normal_obs[:, 1], marker="*", zorder = 10, alpha = 0.6, facecolors='none', edgecolors='black', s = point_size + 20)


        if ('X_grid' in plot_dict) :
            X_grid_for_tri = plot_dict['X_grid']
        else :
            X_grid_for_tri = self.X_domain # filter tri for plot
        X_triang = ptri.Triangulation(X_grid_for_tri[:,0], X_grid_for_tri[:,1])

        if method == 'error_relative' :
            error_norm = np.linalg.norm(error_domain, axis = 1)
            error_max_plot = 0.5

            levels = np.linspace(0, error_max_plot, 100)
            colormap_plot = plt.tricontourf(X_triang, error_norm, levels = levels, cmap = 'brg',
                                        vmin = 0, vmax = error_max_plot )
            
        elif method == 'source_error' :
            emax = source_error_domain.max()
            emin = 0.0
            levels = np.linspace(emin, emax, 150)
            colormap_plot = plt.tricontourf(X_triang, source_error_domain, levels = levels, cmap = 'brg',
                                        vmin = emin, vmax = emax )
            
        elif method == 'total_SD' :
            vmax_plot = UQ_velocity_interpolation_domain_trace.max()*1.1
            levels = np.linspace(0, vmax_plot, 100)
            colormap_plot = plt.tricontourf(X_triang, UQ_velocity_interpolation_domain_trace, alpha = 0.5, cmap = 'viridis',
                                            vmin = 0, vmax = vmax_plot, levels = levels )

        elif method == 'scalar_variance' :
            if velocity_scalar_variance.shape[0] != self.N_domain :
                X_triang = ptri.Triangulation(self.X_domain_UQ[:,0], self.X_domain_UQ[:,1])

            if 'levels' in kwargs :
                levels = kwargs.get('levels')
                emax_plot = levels[-1]
            else :
                emax_plot = velocity_scalar_variance.max()
                levels = np.linspace(0, emax_plot, 150)

            if 'get_scales' in kwargs and kwargs.get('get_scales') :
                out_dict = { 'levels' : levels }

            colormap_plot = plt.tricontourf(X_triang, velocity_scalar_variance, alpha = 0.5, cmap = 'viridis',
                                            vmin = 0, vmax = emax_plot, levels = levels )


        elif method.endswith('error') :

            emax = kwargs.get('plot_max')
            emin = kwargs.get('plot_min')
            levels = np.linspace(emin, emax, 150)
            colormap_plot = plt.tricontourf(X_triang, field_error, levels = levels, cmap = 'brg', vmin = emin, vmax = emax )

        elif method == 'UQ_field' :

            emax = kwargs.get('plot_max')
            emin = kwargs.get('plot_min')
            levels = np.linspace(emin, emax, 150)
            colormap_plot = plt.tricontourf(X_triang, field_UQ, levels = levels, cmap = 'viridis', vmin = emin, vmax = emax )


        else :
            if method == 'interpolation' :
                velocity_plot_domain = velocity_interpolation_domain
            elif method == 'data' :
                velocity_plot_domain = velocity_data_domain

            # colormap
            norm_velocity_domain = np.linalg.norm(velocity_plot_domain, axis = 1)
            if method == 'data' :
                vmax_plot = norm_velocity_domain.max()*1.1
                self.plot_max_velocity = vmax_plot
                self.internal_data.plot_max_velocity = self.plot_max_velocity

            elif method == 'interpolation' or method == 'u_residual' :
                try : 
                    vmax_plot = self.plot_max_velocity
                except :
                    vmax_plot = norm_velocity_domain.max()*1.1

            if hasattr(self, 'plot_u_max'): # only for velocity
                vmax_plot = self.plot_u_max

            vmin_plot = 0
            levels = np.linspace(vmin_plot, vmax_plot, 200)
            if not hasattr(self, 'v_map') :
                self.v_map = 'rainbow' # default
            colormap_plot = plt.tricontourf(X_triang, norm_velocity_domain, levels = levels, alpha = 0.55, cmap = self.v_map,
                                        vmin = vmin_plot, vmax = vmax_plot, antialiased=False    )

        object = plt.fill(self.X_obstacle_visu[:,0], self.X_obstacle_visu[:,1], 0.8, color='gray')

        # format
        if 'title' in kwargs :
            plt.title(kwargs.get('title'))
        else :
            plt.title(f'Field : {method}')

        plt.axis('equal')
        plt.ylim(self.domain[1,0], self.domain[1,1])
        plt.xlim(self.domain[0,0], self.domain[0,1])

        fontsize = 20

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        if not self.config.hide_colorbar :
            cb_ax = fig.add_axes([0.91, 0.124, 0.02, 0.754])
            if np.abs(colormap_plot.norm.vmax + colormap_plot.norm.vmin)/2 < 1e-2 :
                colorbar_format = plticker.FormatStrFormatter('%.2e')
            elif np.abs(colormap_plot.norm.vmax + colormap_plot.norm.vmin)/2 < 1.0 :
                colorbar_format = plticker.FormatStrFormatter('%.3f')
            else :
                colorbar_format = plticker.FormatStrFormatter('%.2f')
            cbar = plt.colorbar(colormap_plot, cax = cb_ax, format = colorbar_format)
            cbar.ax.tick_params(labelsize = fontsize)

        # save plot
        
        if 'save_name' in kwargs :
            save_name = kwargs.get('save_name') + '.pdf'
        elif method == 'error_relative' :
            save_name = 'relative_error.pdf'
        elif method == 'source_error' :
            save_name = 'source_error.pdf'
        elif method == 'interpolation' :
            save_name = 'velocity_interpolation.pdf'
            if hasattr(self, 'X_obs') :
                print(f'[Observation] Design direct observations size : {self.X_obs.shape[0]}')
            if hasattr(self, 'X_normal_obs') :
                print(f'[Observation] Design discrete BC observations size {self.X_normal_obs.shape[0]}' )
        elif method == 'data' :
            save_name = 'data_velocity.pdf'
        elif method == 'total_SD' :
            save_name = 'sqrt_trace_variance.pdf'
        elif method == 'scalar_variance':
            save_name = 'scalar_variance_field.pdf'
        try :
            plt.savefig(r".\BCGP_figures\\" + save_name, bbox_inches='tight', pad_inches=0.01)
        except :
            Warning('[Plot] Set correct path for saving figures')

        if 'out_dict' in locals() :
            return out_dict


    def plot_profile_indicators(self, spectral_precision_grid, rel_agg_obstacle_normal, abs_scalar_stream ) :

        plt.figure()
        plt.semilogy(spectral_precision_grid, rel_agg_obstacle_normal, marker='o', linestyle='-', c = 'black', markerfacecolor='white', markeredgecolor='black', label = 'Normal indicator')
        plt.semilogy(spectral_precision_grid, np.array(abs_scalar_stream).mean(axis = 1), '*-', c='black', label = 'Stream indicator')            
        plt.xlabel(r"$-$ Log of spectral expansion precision")
        plt.ylabel("Evaluation of profile indicators", rotation = 90)
        plt.grid(True, which="minor", alpha = 0.6, ls = '--')
        plt.legend(loc='upper right')

        return


    def plot_solution_snapshot(self, **kwargs) :

        if 'time_it' in kwargs :
            plot_velocity = self.velocity_truth[kwargs.get('time_it'),:]
        else :
            plot_velocity = self.velocity_truth

        plot_dict = {   'velocity_data_domain'  :   plot_velocity   }

        self.do_plot(plot_dict, method = 'data') # use same X_domain as interpolation for comparison

    def print_title(self, text) :
        width = 110
        print(f"{'-' * ((width - len(text)))}{text}")

    def print_subtitle(self, text) :
        width = 90
        print(f"{'-' * ((width - len(text)))}{text}")




