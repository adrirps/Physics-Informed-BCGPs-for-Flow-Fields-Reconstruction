
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt

# Profile boundary computations (cylinder and NACA 4-digit airfoils)

class cylinder(object) :

    def __init__(self, cylinder_parameters):

        x0 = cylinder_parameters[0]
        y0 = cylinder_parameters[1]
        radius = cylinder_parameters[2]

        s = smp.symbols('s')

        gamma_1_smp = x0 + radius*smp.cos(s)
        gamma_2_smp = y0 + radius*smp.sin(s)

        self.gamma_1_np = smp.lambdify([s], gamma_1_smp, 'numpy')
        self.gamma_2_np = smp.lambdify([s], gamma_2_smp, 'numpy')
        

class NACA_airfoil(object):
    
    def __init__(self,NACA_code):

        self.NACA_code = NACA_code

        digits = [int(digit) for digit in str(NACA_code)]
        M = digits[0]/100
        P = digits[1]/10
        T = (digits[2] + digits[3]/10)/10

        s = smp.symbols('s')

        # x change of variable
        x = (1 + smp.cos(s))/2

        # NACA standard coefficients

        a0 = 0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = 0.2843
        # a4 = -0.1015 # open-end
        a4 = -0.1036 # closed-end (default)

        ythick = (5*T*(a0*smp.sqrt(x) + a1*x + a2*(x**2) + a3*(x**3) + a4*(x**4) ))

        ycamber = smp.Piecewise(
            ( (M/(P**2))*((2*P*x)-(x**2)), (0 <= x) & (x < P) ),
            ( (M/((1-P)**2))*(1-(2*P)+(2*P*x)-(x**2)), (P <= x) & (x <= 1) ) )

        dx_ycamber = smp.Piecewise(
            ( (2*M / P**2)*(P - x), (0 <= x) & (x < P) ),
            ( (2*M/((1 - P)**2))*(P - x), (P <= x) & (x <= 1) ) )

        # NACA coordinates

        x_NACA = smp.Piecewise( (x - ythick*dx_ycamber/smp.sqrt(1 + (dx_ycamber**2)), s <= smp.pi ),
                                (x + ythick*dx_ycamber/smp.sqrt(1 + (dx_ycamber**2)), s > smp.pi ) )
        y_NACA = smp.Piecewise( (ycamber + ythick/smp.sqrt(1 + (dx_ycamber**2)), s <= smp.pi ),
                                (ycamber - ythick/smp.sqrt(1 + (dx_ycamber**2)), s > smp.pi ) )

        # curve derivatives

        ds_x_NACA = smp.diff(x_NACA, s)
        ds_y_NACA = smp.diff(y_NACA, s)

        # numpy functions

        self.x_NACA_np = smp.lambdify([s], x_NACA, 'numpy')
        self.y_NACA_np = smp.lambdify([s], y_NACA, 'numpy')
        self.dyc_np = smp.lambdify([s], dx_ycamber, 'numpy')
        self.ds_x_NACA_np = smp.lambdify([s], ds_x_NACA, 'numpy')
        self.ds_y_NACA_np = smp.lambdify([s], ds_y_NACA, 'numpy')


    def boundary_values(self,N_grid,c_length, limits) :

        # Airfoil is computed for yc sapnning from 0 to 1 * c_length

        s_series = np.linspace(limits[0],limits[1],N_grid)
        if s_series[0] < 0 : # order periodic elements
            s_series[s_series < 0] += 2*np.pi
        val_out = np.zeros((N_grid,2))
        val_out[:,0] = self.x_NACA_np(s_series)
        val_out[:,1] = self.y_NACA_np(s_series)
        val_out = c_length*val_out

        return val_out, s_series
    
    def curve_derivatives_values(self,N_grid,c_length, **kwargs) :

        # Airfoil is computed for yc sapnning from 0 to 1 * c_length

        if 's_series' in kwargs :
            s_series = kwargs.get('s_series')
            N_grid = len(s_series)
        else :
            s_series = np.linspace(0,2*np.pi,N_grid) # over [0,2 pi]

        val_out = np.zeros((N_grid,2))
        default_set = np.seterr(divide='ignore', invalid='ignore') # warning avoid
        val_out[:,0] = self.ds_x_NACA_np(s_series)
        val_out[:,1] = self.ds_y_NACA_np(s_series)
        np.seterr(**default_set) # warning defaults
        val_out = c_length*val_out
        val_norm = np.linalg.norm(val_out, axis = 1)

        # filtering indexes
        if 'not_filter' in kwargs and kwargs.get('not_filter') :
            pass
        else :
            nan_indexes = np.isnan(val_norm)
            zero_norm_indexes = val_norm < 1e-12 # small numeric tolerance
            filter_indexes = ( ~ nan_indexes) * ( ~ zero_norm_indexes)
            val_out = val_out[filter_indexes,:]
            val_norm = val_norm[filter_indexes]
        
            if np.sum( ~ filter_indexes) > 0 :
                print(f'[Airfoil] Avoiding {np.sum( ~ filter_indexes)} zero/nan normal vectors at profile')

        if 'normalized' in kwargs and kwargs.get('normalized') :
            val_out[:,0] = val_out[:,0] / val_norm
            val_out[:,1] = val_out[:,1] / val_norm

        if 'series_out' in kwargs and kwargs.get('series_out') :
            return val_out, s_series

        if 'not_filter' in kwargs and kwargs.get('not_filter') :
            return val_out
        
        return val_out, filter_indexes


    def plot_airfoil(self,x, y) :

        plt.figure()
        plt.plot(x,y,'-')
        plt.plot(x[-1],y[-1],'o')
        plt.axis('equal')
        plt.title(F'NACA Airfoil {self.NACA_code}')


