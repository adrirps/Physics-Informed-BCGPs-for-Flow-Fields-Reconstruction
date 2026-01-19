
import numpy as np
import time

from .object_tools import main_tools


# Model definitions

# -------------------------------------------------------------------------------------------------------------
# GPR from data (Eulerian data)
# -------------------------------------------------------------------------------------------------------------

class GPR_model(main_tools):

    def __init__(self, input_config):
        super().__init__(input_config)


    def perform_GPR(self, GP, observations_dict, **kwargs) :

        text = 'Computing GPR'
        print(f"{'-' * ((80 - len(text)))}{text}")

        out_dict = {}

        # Mean computation

        start_time_mean = time.time()

        if 'X_particular' in kwargs :

            X_particular = kwargs.get('X_particular')

            if not ('out_list' in kwargs) :
                velocity_particular = GP.interpolation('velocity_mean', X_particular, observations_dict).reshape((X_particular.shape[0], 2))
                out_dict['velocity_particular'] = velocity_particular
            else:
                out_list = kwargs.get('out_list')

                if 'velocity' in out_list :
                    velocity_particular = GP.interpolation('velocity_mean', X_particular, observations_dict).reshape((X_particular.shape[0], 2))
                    out_dict['velocity_particular'] = velocity_particular


        else:
            out_list = kwargs.get('out_list')

            if 'domain' in out_list :
                velocity_interpolation_domain = GP.interpolation('velocity_mean', self.X_domain, observations_dict).reshape((self.N_domain,2))
                out_dict['u_domain'] = velocity_interpolation_domain

            if 'obstacle' in out_list :
                velocity_interpolation_obstacle = GP.interpolation('velocity_mean', self.X_obstacle, observations_dict).reshape((self.N_obstacle,2))
                out_dict['obstacle'] = velocity_interpolation_obstacle

            if 'obs_check' in out_list :
                out_dict['obs_check'] = GP.interpolation('velocity_mean', observations_dict['points'], observations_dict).reshape((observations_dict['points'].shape[0],2))

            if 'stream' in out_list :
                scalar_stream_domain = GP.interpolation('scalar_stream_mean', self.X_domain, observations_dict)
                out_dict['stream'] = scalar_stream_domain

            if 'stream_obstacle' in out_list :
                scalar_stream_obstacle = GP.interpolation('scalar_stream_mean', self.X_obstacle, observations_dict)
                out_dict['stream_obstacle'] = scalar_stream_obstacle

            if 'UQ' in out_list :

                start_time_cov = time.time()

                UQ_velocity_interpolation_domain = GP.interpolation('velocity_covariance', self.X_domain, observations_dict)
                UQ_velocity_interpolation_domain_trace = np.sqrt( np.diag(UQ_velocity_interpolation_domain[::2, ::2] + UQ_velocity_interpolation_domain[1::2, 1::2]) )

                end_time_cov = time.time()

                print(f"Execution covariance interpolation time: {end_time_cov - start_time_cov:.6f} seconds")

        end_time_mean = time.time()

        print(f"Computation (and save) time of mean estimates: {end_time_mean - start_time_mean:.3f} seconds")

        # Uncertainty computation

        if ('out_list' in kwargs) and ('UQ' in kwargs.get('out_list')) :
            start_time_cov = time.time()
            UQ_velocity_interpolation_domain = GP.interpolation('velocity_covariance', self.X_domain, observations_dict)

            UQ_velocity_interpolation_domain_trace = np.sqrt( np.diag(UQ_velocity_interpolation_domain[::2, ::2] + UQ_velocity_interpolation_domain[1::2, 1::2]) )

            end_time_cov = time.time()

            out_dict['UQ_domain_trace'] = UQ_velocity_interpolation_domain_trace

            print(f"Execution covariance interpolation time: {end_time_cov - start_time_cov:.3f} seconds")

        # Plots

        if hasattr(self.config, 'visualize') and self.config.visualize and ('velocity_interpolation_domain' in locals()) :
            
            plot_dict = {
                'velocity_interpolation_domain'             :   velocity_interpolation_domain
            }

            if 'plot_list' in kwargs :
                plot_list = kwargs.get('plot_list')
                if 'total_SD' in plot_list :
                    plot_dict['UQ_velocity_interpolation_domain_trace'] = UQ_velocity_interpolation_domain_trace
                    self.do_plot(plot_dict, method = 'total_SD')
                if 'velocity' in plot_list :
                    self.do_plot(plot_dict, method = 'interpolation')

        return out_dict




