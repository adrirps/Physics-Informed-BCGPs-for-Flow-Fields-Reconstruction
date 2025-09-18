import os
import dill
import gzip
import shutil

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as pth
import matplotlib.tri as ptri
import matplotlib.animation as pla
import core.Ofpp as Ofpp
from datetime import datetime

# Utility for OpenFOAM data treatment

class FOAM_for_GPR(object):

    def __init__(self,case_folder_path,**kwargs):

        print('[Data] Loading data from ' + case_folder_path)

        # Read cell centres data files
        # Centers must be included in iteration 0
        # The centers file 'C' can be obtained with 'postProcess -func writeCellCentres -time 0' in case folder using OpenFOAM

        self.config = kwargs.get('config')

        # set time reference
        time_series, time_step = self.set_time(case_folder_path, kwargs.get('time_input'))

        # Read mesh and cell centres
        try :
            centers_file_path = os.path.join(case_folder_path, r'0\C') # read at iteration 0
            foam_mesh_cell_centres = Ofpp.parse_internal_field(centers_file_path)
        except :
            raise ValueError('[Data] There is no centers file C in the case folder - iteration 0')

        foam_mesh_num_cell = foam_mesh_cell_centres.shape[0]

        print(f'[Data] Mesh and {foam_mesh_num_cell} cell centers loaded')

        # Set reference and domain
        foam_centres = self.set_reference(foam_mesh_cell_centres[:,0:2], kwargs.get('reference_point')) # get only 2D data
        domain = self.set_domain(foam_centres, domain_input = kwargs.get('domain_input'))

        # Read velocity data files (all iterations)
        N_time = time_series.shape[0]
        u_data_internal = np.zeros((N_time,foam_mesh_num_cell,2))


        for it_t in range(N_time):
            file_path = os.path.join(case_folder_path, str(time_series[it_t])) # add time
            u_data_internal_read = Ofpp.parse_internal_field(file_path + '\\U')
            u_data_internal[it_t,:,:] = u_data_internal_read[:,0:2]

        self.case_folder_path = case_folder_path
        self.domain = domain
        self.time_series = time_series
        self.time_step = time_step
        self.foam_centres = foam_centres
        self.foam_centres_triang = sp.spatial.Delaunay(self.foam_centres) # for cell identification
        self.foam_centres_tree = sp.spatial.KDTree(self.foam_centres) # for nearest neighbor
        self.u_data_internal = u_data_internal
        self.obstacle_type = self.config.obstacle_type
        self.obstacle_parameters = self.config.obstacle_parameters

        if 'boundary_source_precision' in kwargs :
            self.boundary_source_precision = kwargs.get('boundary_source_precision')
        elif hasattr(self.config, 'boundary_source_precision'):
            self.boundary_source_precision = self.config.boundary_source_precision

        print('[Data] OpenFOAM data loaded !')

    def set_reference(self, centres, reference_point) :
        centres = centres - reference_point
        print(f'[Data] Loading cell centres with reference shift {reference_point}')
        return centres
    
    def set_domain(self,centres, domain_input) :
        centres_min = centres.min(axis = 0)
        centres_max = centres.max(axis = 0)
        domain = np.vstack((centres_min,centres_max)).transpose()
        if domain_input is not None :
            domain[:,0] = np.maximum(domain[:,0],domain_input[:,0])
            domain[:,1] = np.minimum(domain[:,1],domain_input[:,1])
        print(f'[Data] Domain reference set for data as to [{domain[0,0]}, {domain[0,1]}] x [{domain[1,0]}, {domain[1,1]}]')
        return domain
    
    def set_time(self,case_folder_path,time_input) :

        all_files = np.array(os.listdir(case_folder_path))
        isdigit_vect = np.vectorize(str.isdigit)
        digit_files = all_files[isdigit_vect(all_files)]

        # Reorder time and set interval
        time_series = digit_files.astype(int)
        ordered_time_indexes = np.argsort(time_series)
        time_series = time_series[ordered_time_indexes][1:] # file '0' is not data but OpenFoam configuration

        if time_input is not None :
            input_indexes = (time_series >= time_input[0])*(time_series <= time_input[1])
            time_series = time_series[input_indexes]

        print(f'[Data] Time interval set to [{time_series[0]}, {time_series[-1]}]')

        time_step = float(time_series[1]-time_series[0]) # supposed uniform

        return time_series, time_step
    
    def read_velocity_compressed_file(file_path) :

        try :
            with gzip.open(file_path + "\\U.gz", "rb") as f_in:
                with open("U", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return Ofpp.parse_internal_field(file_path + '\\U')
        except:
            raise ValueError('[Data] Reading error of velocity file U.gz')


    def compute_boundary_section(self, boundary_type, method, **kwargs):

        print(f'[Data] Computing boundary data ({boundary_type})')

        N_time = len(self.time_series)

        if method == 'all_data_points' :

            if boundary_type == 'inlet' or boundary_type == 'outlet' :
                if 'target' in kwargs :
                    horizontal_target = kwargs.get('target')
                elif boundary_type == 'inlet' :
                    horizontal_target = self.domain[0,0]
                elif boundary_type == 'outlet' :
                    horizontal_target = self.domain[0,1]

                if hasattr(self, 'boundary_source_precision') :
                    precision = self.boundary_source_precision
                else :
                    precision = min(np.abs(self.foam_centres[:,0] - horizontal_target))
                indexes = np.abs(self.foam_centres[:, 0] - horizontal_target) <= precision

                X_boundary = self.foam_centres[indexes,:]
                velocity_boundary = self.u_data_internal[:,indexes,:]
                    
                # verify it is only a single column
                if self.obstacle_type != 'NACA_airfoil' :
                    values, layer_counts = np.unique(X_boundary[:, 0], return_counts = True)
                    if values.shape[0] > 1 :
                        if boundary_type == 'inlet' :
                            indexes = X_boundary[:,0] == values.min()
                        elif boundary_type == 'outlet' :
                            indexes = X_boundary[:,0] == values.max()
                        X_boundary = X_boundary[indexes, :] 
                        velocity_boundary = velocity_boundary[:,indexes,:]
                
            elif boundary_type == 'top' or boundary_type == 'bottom' :
                if 'target' in kwargs :
                    vertical_target = kwargs.get('target')
                elif boundary_type == 'top' :
                    vertical_target = self.domain[1,1]
                elif boundary_type == 'bottom' :
                    vertical_target = self.domain[1,0]

                if hasattr(self, 'boundary_source_precision') :
                    precision = self.boundary_source_precision
                else :
                    precision = min(np.abs(self.foam_centres[:,1] - vertical_target))*3
                indexes = np.abs(self.foam_centres[:, 1] - vertical_target) <= precision

                X_boundary = self.foam_centres[indexes,:]
                velocity_boundary = self.u_data_internal[:,indexes,:]


        else : # Other methods : nearest neighbor over N_boundary points // data interpolation

            N_boundary = kwargs.get('N_boundary')
            velocity_boundary = np.zeros((N_time,N_boundary,2))

            # set a priori boundary points
            if boundary_type == 'inlet' or boundary_type == 'outlet' :
                if 'without_limits' in kwargs and kwargs.get('without_limits') :
                    X1 = np.linspace(self.domain[1,0],self.domain[1,1], N_boundary + 2 )[1:-1]
                else :
                    X1 = np.linspace(self.domain[1,0],self.domain[1,1],N_boundary)
                
                if boundary_type == 'inlet' :
                    X0 = np.repeat(self.domain[0,0],(N_boundary,))
                elif boundary_type == 'outlet':
                    X0 = np.repeat(self.domain[0,1],(N_boundary,))

            elif boundary_type == 'top' or boundary_type == 'bottom' :
                if 'without_limits' in kwargs and kwargs.get('without_limits') :
                    X0 = np.linspace(self.domain[0,0],self.domain[0,1], N_boundary + 2 )[1:-1]
                else :
                    X0 = np.linspace(self.domain[0,0],self.domain[0,1],N_boundary)
                
                if boundary_type == 'top' :
                    X1 = np.repeat(self.domain[1,1],(N_boundary,))
                elif boundary_type == 'bottom':
                    X1 = np.repeat(self.domain[1,0],(N_boundary,))
                
            X_boundary_search = np.column_stack((X0,X1))

            # get boundary values

            if method == 'nearest' :
                # search for data points
                distances, nearest_indexes = self.foam_centres_tree.query(X_boundary_search)
                X_boundary = self.foam_centres[nearest_indexes]
                N_boundary = X_boundary.shape[0]

                # set velocity values
                N_time = len(self.time_series)
                velocity_boundary = np.zeros((N_time, N_boundary, 2))
                velocity_boundary = self.u_data_internal[:, nearest_indexes, :]

                print(f'[Data] Boundary points for {boundary_type} section set according to nearest neighbor')

            else :
                
                X_boundary = X_boundary_search

                # get triangular convex hull
                indexes = self.foam_centres_triang.find_simplex(X_boundary)
                nearest_cell_centres_indexes = self.foam_centres_triang.simplices[indexes]
                nearest_cell_centres = self.foam_centres[nearest_cell_centres_indexes]

                for it_time in range(N_time) :
                    
                    nearest_cell_velocities = self.u_data_internal[it_time,:,:][nearest_cell_centres_indexes]
                    for it_x in range(N_boundary) :

                        do_plot = False
                        # optional plot
                        if do_plot :
                            fig = plt.figure()
                            ax = fig.add_subplot(111)
                            ax.set_aspect('equal', adjustable='box')
                            ax.scatter(self.foam_centres[:,0], self.foam_centres[:,1])
                            ax.scatter(X_boundary[:,0], X_boundary[:,1])

                            plt.plot(nearest_cell_centres[it_x,:,0], nearest_cell_centres[it_x,:,1], 'o',color = 'orange', markersize=5,zorder=10)
                            plt.plot(X_boundary[it_x,0], X_boundary[it_x,1], 'x',color = 'red', markersize=5,zorder=10)

                        # compute velocity
                        velocity_boundary[it_time,it_x,0] = sp.interpolate.griddata(nearest_cell_centres[it_x,:,:], nearest_cell_velocities[it_x,:,0], X_boundary[it_x,:], method = method)
                        velocity_boundary[it_time,it_x,1] = sp.interpolate.griddata(nearest_cell_centres[it_x,:,:], nearest_cell_velocities[it_x,:,1], X_boundary[it_x,:], method = method)

        return X_boundary, velocity_boundary


    def set_domain_reference(self,ax) : # for plots
        d00 = self.domain[0,0]
        d01 = self.domain[0,1]
        d10 = self.domain[1,0]
        d11 = self.domain[1,1]
        dxh = d01 - d00
        dyh = d11 - d10
        d_gap = 0.2
        ax.set_xlim(d00 - d_gap*dxh, d01 + d_gap*dxh)
        ax.set_ylim(d10 - d_gap*dyh, d11 + d_gap*dyh)
        ax.set_aspect('equal', adjustable='box')
        square = pth.Rectangle((d00, d10), d01, d11, edgecolor='black', facecolor='none')
        ax.add_patch(square)

    def get_domain_points(self, **kwargs) :
        
        X_grid_particular = kwargs.get('X_grid_particular')

        try :
            X_domain_search = np.vstack((X_grid_particular, X_domain_search))
            print('[Data] Points to search at sampled domain and particular grid')
        except :
            X_domain_search = X_grid_particular

        # filtering on obstacle
        X_domain_search = self.filter_domain_obstacle(X_domain_search)

        # search for data points
        distances, nearest_indexes = self.foam_centres_tree.query(X_domain_search)
        X_domain_all = self.foam_centres[nearest_indexes]
        N_domain = X_domain_all.shape[0]

        # set velocity values

        N_time = len(self.time_series)
        
        velocity_domain_all = np.zeros((N_time, N_domain, 2))
        velocity_domain_all = self.u_data_internal[:, nearest_indexes, :]

        # plot
        X_dict = {
            'domain' : X_domain_all,
            'domain search' : X_domain_search
        }
        velocity_dict = {'domain' : velocity_domain_all[3,] }
        symbols_dict = {'domain search' : 'x'}


        if self.config.visualize :
            self.plot_data(X_dict, velocity_dict, symbols_dict = symbols_dict,
                            show_obstacle = True,
                            title = 'Selected domain points')

        X_domain = {
            'all' : X_domain_all
        }

        velocity_domain = {
            'all' : velocity_domain_all
        }

        return X_domain, velocity_domain


    def filter_close_points(X_grid, tol_dist) :
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



    def plot_data(self, X_dict, velocity_dict, **kwargs) :

        if 'symbols_dict' in kwargs :
            symbols_dict = kwargs.get('symbols_dict')
        else :
            symbols_dict = {}

        plot_scale = 51

        plt.figure()

        first_key = True
        for point_key in X_dict :
            if point_key in symbols_dict :
                temp_symbol = symbols_dict[point_key]
            else :
                temp_symbol = 'o'
            if not 'colormap' in kwargs :

                plt.plot(X_dict[point_key][:,0], X_dict[point_key][:,1], temp_symbol)
            
            if point_key in velocity_dict :
                if 'colormap' in kwargs :
                    if kwargs.get('colormap') == 'velocity_norm' and first_key :
                        
                        # Create a triangulation of the irregular grid
                        X_grid_for_tri = np.vstack((X_dict[point_key], X_dict['boundary']))
                        X_triang = ptri.Triangulation(X_grid_for_tri[:,0], X_grid_for_tri[:,1])
                        # add boundary
                        norm_velocity_domain = np.linalg.norm(np.vstack((velocity_dict[point_key], velocity_dict['boundary'])), axis = 1)

                        v_max = 70

                        levels = np.linspace(0, v_max, 100)

                        contour = plt.tricontourf(X_triang, norm_velocity_domain, levels = levels, alpha = 0.3, cmap = 'rainbow',
                                                  vmin = 0, vmax = v_max )
                        plt.colorbar(contour)
                        first_key = False
                if (kwargs.get('quiver') is None) or kwargs.get('quiver') == True :
                    plt.quiver(X_dict[point_key][:,0], X_dict[point_key][:,1], velocity_dict[point_key][:,0], velocity_dict[point_key][:,1],
                            units='dots', width = 1.6, scale = plot_scale, zorder=5)

        # add mesh
        if ('show_mesh' in kwargs) and kwargs.get('show_mesh') :
            plt.triplot(self.foam_centres[:, 0], self.foam_centres[:, 1], self.foam_centres_triang.simplices, color='black')
        
        # add obstacle
        if ('show_obstacle' in kwargs) and kwargs.get('show_obstacle') :
            if self.obstacle_type == 'NACA_airfoil' :
                pass
            else :
                gamma_x0 = self.obstacle_parameters[0] # center
                gamma_y0 = self.obstacle_parameters[1]
                gamma_radius = self.obstacle_parameters[2]
                object = plt.Circle((gamma_x0, gamma_y0), gamma_radius, color='gray', edgecolor='black')
                plt.gca().add_patch(object)

        # format
        plt.axis('equal')
        plt.xlim(self.domain[0,0], self.domain[0,1])
        plt.ylim(self.domain[1,0], self.domain[1,1])
        
        # add title
        if 'title' in kwargs :
            plt.title(kwargs.get('title'))


    def filter_domain_obstacle(self,X_domain):
        
        if self.obstacle_type == 'NACA_airfoil':
            pass

        elif self.obstacle_type == 'cylinder' :

            gamma_x0 = self.obstacle_parameters[0] # center
            gamma_y0 = self.obstacle_parameters[1]
            gamma_radius = self.obstacle_parameters[2]

            points_distance = (X_domain[:,0] - gamma_x0)**2 + (X_domain[:,1] - gamma_y0)**2
            X_domain = X_domain[points_distance > (gamma_radius)**2 ,:]

        print('[Points] Domain points filtered according to obstacle boundary constraint')

        return X_domain

