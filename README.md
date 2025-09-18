## Physics-Informed Boundary-constrained GPs for Flow Fields Reconstruction

This is the repository for the paper [[Padilla-Segarra, Noble, Roustant, Savin: arXiv.2507.17582]](https://doi.org/10.48550/arXiv.2507.17582)

The following commands were used for obtaining reported results presented in Section 5.

For simulations of flow around the cylinder profile:
- For Fig. 4-a: `sim_cylinder_outer_boundary_GPR_velocity_UQ.py --visualize --plot_estimate 'velocity'`
- For Fig. 5-a: `sim_cylinder_outer_boundary_GPR_velocity_UQ.py --visualize --plot_estimate 'total_SD'`
- For Fig. 5-b: `sim_cylinder_outer_boundary_GPR_velocity_UQ.py --visualize --plot_estimate 'total_SD' --base_kernel_without_BC`
- Fig. 6, Fig. 8-a and c: `sim_cylinder_GPR_velocity.py --visualize`
- Fig. 8-b and d: `sim_cylinder_GPR_velocity.py --visualize --base_kernel_without_BC`
- Fig. 7: `sim_cylinder_GPR_velocity.py --spectral_precision 'grid'`

For simulations of flow around the NACA 0412 airfoil:
- Fig. 10-a, b and d:  `sim_NACA_0412_GPR_velocity.py --visualize`
- Fig. 10-c and e : `sim_NACA_0412_GPR_velocity.py --visualize --base_kernel_without_BC`
- Fig. 11-a: `sim_NACA_0412_GPR_velocity.py --KL_measure 'pushforward' --spectral_precision 'grid'`
- Fig. 11-b: `sim_NACA_0412_GPR_velocity.py --KL_measure 'surface' --spectral_precision 'grid'`

The data for the flow simulation around the NACA airfoil was obtained using OpenFOAM by adapting the configuration presented in the raw code of the library AirfRANS (see [AirfRANS Documentation](https://airfrans.readthedocs.io) and [AirfRANS Github](https://github.com/Extrality/AirfRANS)) for consideration of the slip condition on the airfoil boundary.

### Library requirements

- This code uses the symbolic computation package [Sympy](https://www.sympy.org) (version 1.13.1) for computing kernel derivatives and functionals with high precision.

- For parsing OpenFOAM data, we use a modified version of the library [Ofpp](https://github.com/xu-xianghua/ofpp), available in the core folder.



The presentation and architecture of this repository was inspired by [NonLinPDEs-GPsolver](https://github.com/yifanc96/NonLinPDEs-GPsolver).
