## Physics-Informed Boundary-constrained GPs for Flow Fields Reconstruction

This is the repository for the paper [[Padilla-Segarra, Noble, Roustant, Savin: arXiv.2507.17582]](https://doi.org/10.48550/arXiv.2507.17582)

The following commands were used for obtaining reported results presented in Section 5.

For simulations of flow around the cylinder profile:

- Using the base kernel:
    - Fig. 6-b: `sim_cylinder_outer_boundary_GPR_velocity_UQ.py --base_kernel_without_BC`
    - Fig. 9-b and d: `sim_cylinder_GPR_velocity.py --visualize --base_kernel_without_BC`

- Using the BCGP kernel:
    - Fig. 5-a and 6-a: `sim_cylinder_outer_boundary_GPR_velocity_UQ.py`
    - Fig. 7, Fig. 9-a and c: `sim_cylinder_GPR_velocity.py --visualize`
    - Fig. 8: `sim_cylinder_GPR_velocity.py --spectral_precision 'grid'`

- Sensibility analysis of the normal fit indicator (the script is automatically re-run for comparison with the BCGP kernel):
    - Fig. 10: `sim_cylinder_GPR_velocity_sensibility.py --base_kernel_without_BC`

For simulations of flow around the NACA 0412 airfoil:

- Hyperparameter estimation with UQ-based cross validation:
    - Fig. 12-a:  `sim_NACA_0412_GPR_parameter_kCV.py --without_div_free --additive_scales 0 --base_kernel_without_BC`
    - Fig. 12-b:  `sim_NACA_0412_GPR_parameter_kCV.py --additive_scales 0 --base_kernel_without_BC`
    - Fig. 12-c:  `sim_NACA_0412_GPR_parameter_kCV.py --additive_scales 3 --base_kernel_without_BC`
    - Fig. 12-d:  `sim_NACA_0412_GPR_parameter_kCV.py --additive_scales 3`

- Using the RBF kernel:
    - Fig. 13-a to c, Fig. 14-a, Fig. 15-a and e:  `sim_NACA_0412_GPR_velocity.py --visualize --kernel_parameter 0.2 0.1 8 0 --without_div_free --base_kernel_without_BC`
- Using the divergence-free (DF-RBF) kernel:
    - Fig. 13-d to f, Fig. 14-b, Fig. 15-b and f:  `sim_NACA_0412_GPR_velocity.py --visualize --kernel_parameter 0.01 0.1 8 0 --base_kernel_without_BC`
- Using the multi-scale (M-RBF) kernel:
    - Fig. 13-g to i, Fig. 14-c, Fig. 15-c and g:  `sim_NACA_0412_GPR_velocity.py --visualize --kernel_parameter 0.05 1.0 8 3 --base_kernel_without_BC`
- Using the physics-informed boundary-constrained (PI-RBF) kernel:
    - Fig. 13-j to l, Fig. 14-d, Fig. 15-d and h:  `sim_NACA_0412_GPR_velocity.py --visualize --kernel_parameter 0.1 1.0 8 3`
    - Fig. 16-a: `sim_NACA_0412_GPR_velocity.py --kernel_parameter 0.1 1.0 8 3 --KL_measure 'pushforward' --spectral_precision 'grid'`
    - Fig. 16-b: `sim_NACA_0412_GPR_velocity.py --kernel_parameter 0.1 1.0 8 3 --KL_measure 'surface' --spectral_precision 'grid'`

The data for the flow simulation around the NACA airfoil was obtained using OpenFOAM by adapting the configuration presented in the raw code of the library AirfRANS (see [AirfRANS Documentation](https://airfrans.readthedocs.io) and [AirfRANS Github](https://github.com/Extrality/AirfRANS)) for consideration of the slip condition on the airfoil boundary.

### Library requirements

- This code uses the symbolic computation package [Sympy](https://www.sympy.org) (version 1.13.1) for computing kernel derivatives and functionals with high precision.

- For parsing OpenFOAM data, we use a modified version of the library [Ofpp](https://github.com/xu-xianghua/ofpp), available in the core folder.



The presentation and architecture of this repository was inspired by [NonLinPDEs-GPsolver](https://github.com/yifanc96/NonLinPDEs-GPsolver).
