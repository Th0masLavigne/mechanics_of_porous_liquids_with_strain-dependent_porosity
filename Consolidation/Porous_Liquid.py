# Date: 20/11/2025
# Author: Thomas Lavigne
# Reference: Giuseppe Sciume
# Laboratory: I2M
#
# --------------------------------------- #
# 			Import libraries			  #
# --------------------------------------- #
import sys
import os
import dolfinx
import mpi4py
import ufl
import basix
import numpy
import time
import pandas
from dolfinx.fem.petsc import NonlinearProblem
#
if mpi4py.MPI.COMM_WORLD.rank == 0:
	print("Dolfinx version is:",dolfinx.__version__)
#
# Compute the overall computation time
# Set time counter
begin_t = time.time()
#
# --------------------------------------- #
# 	   Import user-defined functions	  #
# --------------------------------------- #
# Add path to the service files
sys.path.append('./service_files_consolidation')
#
import geometry_porous_liquid
import constitutive_laws
import functions
#
# --------------------------------------- #
# 	    More user-defined functions	      #
# --------------------------------------- #
#
def terzaghi_p(x):
	"""
	Compute the Exact Terzaghi solution for L2 error estimation
	Inputs: coordinates of the mesh
	outputs_fluid_terzaghi: Fluid pressure
	Note: the elapsed time is taken relative to TPHASE1, the start of the
	loaded phase, following the two-phase loading protocol of the new
	formulation. This function is only meaningful once the load is applied
	(t >= TPHASE1), which is enforced by only calling it during Phase 2.
	"""
	# applied pressure and characteristic size of the domain
	p0,L = p_left_terzaghi, length_cylinder
	# initialize pression
	pression = 0
	# number of iterations
	kmax=1e3
	# elapsed time since the load was applied
	t_rel = t - TPHASE1
	# Consider coordinate (L-x) to respect Terzaghi standards and compute the series
	for k in range(1,int(kmax)):
		pression += 4/numpy.pi*(-1)**(k-1)/(2*k-1)*numpy.cos((2*k-1)*0.5*numpy.pi*((L-x[0])/L))*numpy.exp(-(2*k-1)**2*0.25*numpy.pi**2*cv*t_rel/L**2)
		pl = p0*pression
	return pl
#
def terzaghi_v(x):
	"""
	Compute the Exact Terzaghi solution for L2 error estimation
	Inputs: coordinates of the mesh
	Outputs: Fluid pressure
	Note: the elapsed time is taken relative to TPHASE1, see terzaghi_p above.
	"""
	# applied pressure and characteristic size of the domain
	p0,L = p_left_terzaghi, length_cylinder
	# initialize displacement
	speed = 0
	# number of iterations
	kmax=1e3
	# elapsed time since the load was applied
	t_rel = t - TPHASE1
	# Consider coordinate (L-x) to respect Terzaghi standards and compute the series
	for k in range(1,int(kmax)):
		speed += 2/(L*M)*(-1)**(k+1)*cv*numpy.sin((2*k-1)*0.5*numpy.pi*((L-x[0])/L))*numpy.exp(-(2*k-1)**2*0.25*numpy.pi**2*cv*t_rel/L**2)
		speeds = p0*speed
	return speeds
#
def L2_error_p(mesh, P1, __p):
    P1space = dolfinx.fem.functionspace(mesh, P1)
    p_theo  = dolfinx.fem.Function(P1space)
    p_theo.interpolate(terzaghi_p)
    L2_errorp = dolfinx.fem.form(ufl.inner(__p - p_theo, __p - p_theo) * dx)
    L2_vol  = dolfinx.fem.form(1 * dx)

    num_local = dolfinx.fem.assemble_scalar(L2_errorp)
    den_local = dolfinx.fem.assemble_scalar(L2_vol)

    num = mesh.comm.allreduce(num_local, op=mpi4py.MPI.SUM)
    den = mesh.comm.allreduce(den_local, op=mpi4py.MPI.SUM)
    return numpy.sqrt(num / den)
#
def L2_error_v(mesh, P1, __v):
    P1space = dolfinx.fem.functionspace(mesh, P1)
    v_theo  = dolfinx.fem.Function(P1space)
    v_theo.interpolate(terzaghi_v)
    L2_errorv = dolfinx.fem.form(ufl.inner(__v.sub(0) - v_theo, __v.sub(0) - v_theo) * dx)
    L2_vol  = dolfinx.fem.form(1 * dx)

    num_local = dolfinx.fem.assemble_scalar(L2_errorv)
    den_local = dolfinx.fem.assemble_scalar(L2_vol)

    num = mesh.comm.allreduce(num_local, op=mpi4py.MPI.SUM)
    den = mesh.comm.allreduce(den_local, op=mpi4py.MPI.SUM)
    return numpy.sqrt(num / den)
#
def evaluate_point(mesh, function, contributing_cells, point, output_list, index):
    """
    Parallel-safe version of point evaluation, compatible with the original interface.

    Inputs:
        mesh              : dolfinx mesh
        function          : dolfinx.fem.Function to evaluate
        contributing_cells: list/array of local cell indices that might contain the point
        point             : coordinates of the point (1D array of length mesh.geometry.dim)
        output_list       : pre-allocated Python list or array to store the value
        index             : index in output_list to store the result
    Actions:
        Evaluates `function` at `point` if the current processor owns the cell,
        gathers the result to rank 0, and sets `output_list[index]`.
    """
    val_local = None

    # Only evaluate if the point is contained in a local cell
    if len(contributing_cells) > 0:
        try:
            val_local = function.eval(point, [int(contributing_cells[0])])
            val_local = float(val_local[0])  # Convert to scalar if function is scalar
        except Exception:
            val_local = None

    # Gather values from all ranks
    gathered_vals = mesh.comm.gather(val_local, root=0)

    # Rank 0 chooses the first non-None value and stores it in output_list
    if mesh.comm.rank == 0:
        for v in gathered_vals:
            if v is not None:
                output_list[index] = v
                break

    return None
#
# --------------------------------------- #
# 	   		 Time parameters			  #
# --------------------------------------- #
#
interfaces: bool = True


hour_to_sec = 3600 # [s/h]

if interfaces:
	TPHASE1 = 50.0*hour_to_sec
	# Duration of the loaded (consolidation) phase [s]
	TPHASE2 = 50.0*hour_to_sec
else:
	TPHASE1 = 1.0*hour_to_sec
	# Duration of the loaded (consolidation) phase [s]
	TPHASE2 = 21.0*hour_to_sec

# Time step [s]
dt    = 0.005*hour_to_sec

# Number of required steps in each phase [-]
num_steps_phase1 = int(TPHASE1/dt)
num_steps_phase2 = int(TPHASE2/dt)
# Total number of steps (both phases)
num_steps        = num_steps_phase1 + num_steps_phase2
# Total simulated duration, kept for backward compatibility with the
# analytical post-processing below
t_fin = TPHASE1 + TPHASE2
#
# time steps (counted from the start of the loaded phase, TPHASE2) to evaluate the pressure in space:
n0, n1, n2 = 400, 800, 1200 #2h, 4h, 8h
#
# --------------------------------------- #
# 	   	Geometrical parameters			  #
# --------------------------------------- #
#
# Radius of the cylinder [m]
radius_cylinder = 0.01
# length of the cylinder [m]
length_cylinder = 0.1
#
# --------------------------------------- #
# 	   		Generate Mesh				  #
# --------------------------------------- #
# Spacial discretising [-]
Nx = 40
# Create gmsh tagged mesh
gmsh_name = "geometry/2D_axysim_rectangle.msh"
create_new_mesh = False
# Need to regenerate the mesh or not
if (os.path.exists(gmsh_name)) and (create_new_mesh == False):
	pass
else:
	if mpi4py.MPI.COMM_WORLD.rank == 0:
		geometry_porous_liquid.Axysimmetric_rectangle_mesh(length_cylinder, radius_cylinder, Nx, size_min=0.1, size_max=1, log=True, optimise=True, netgen=True)
#
# Read the mesh and the mesh tags
gmsh_model_rank = 0
gdim            = 2
#
domain, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(gmsh_name, mpi4py.MPI.COMM_WORLD, gmsh_model_rank, gdim=gdim)
#
# To ensure proper reading, one can refer to this xdmf including the boundary tags not working in // need to explicit xdmf_tag = ...
xdmftag = dolfinx.io.XDMFFile(mpi4py.MPI.COMM_WORLD, 'outputs_fluid_terzaghi/debug_tags_mesh.xdmf', "w")
xdmftag.write_mesh(domain)
xdmftag.write_meshtags(facet_tags, domain.geometry)
xdmftag.write_meshtags(cell_tags,  domain.geometry)
xdmftag.close()
#
#
mesh_stats = functions.get_element_size_stats(domain)
volume  = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.Measure('dx', domain=domain)))
volume_ = domain.comm.allreduce(volume, op=mpi4py.MPI.SUM)
if mpi4py.MPI.COMM_WORLD.rank == 0:
	print("--- Loaded Mesh Stats ---")
	print(f"Total Volume: {volume_}")
	print(f"Mean Element Size (h_bar): {mesh_stats['mean_h']:.6f}")
	print(f"Standard Deviation (sigma_h): {mesh_stats['std_dev_h']:.6f}")
#
#
# --------------------------------------- #
# 	   		Evaluation points			  #
# --------------------------------------- #
#
# Identify contributing cells to our points of interest for post processing
# Naming convention for the 3 continuously-tracked points (matches the
# legacy reference porous_liquid_2026_all3.py's points_suivi / the
# article's PA/PB/PC labels):
#   PA = drainage surface = x=0      -> cells_x_0, points[2]
#   PB = middle           = x=L/2    -> cells_x_L_over_2, points[1]
#   PC = bottom             = x=L      -> cells_x_L, points[0]
num_points = 15
# Physical points we want an evaluation in
x_check          = numpy.linspace(0,length_cylinder,num_points)
points_for_time  = numpy.array([[length_cylinder, 0., 0.], [length_cylinder/2, 0., 0.]])
points_for_space = numpy.zeros((num_points,3))
for ii in range(num_points):
	points_for_space[ii,1] = 0.
	points_for_space[ii,0] = x_check[ii]
# Create the bounding box tree and identify contributing cells
tree             = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
points           = numpy.concatenate((points_for_time,points_for_space))
cell_candidates  = dolfinx.geometry.compute_collisions_points(tree, points)
colliding_cells  = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points)
cells_x_L        = colliding_cells.links(0)
cells_x_L_over_2 = colliding_cells.links(1)
cells_x_0        = colliding_cells.links(2)
#
# --------------------------------------- #
# 	   		Material parameters			  #
# --------------------------------------- #
#
# Viscosities [Pa.s]
scafflod_like_viscosity      = dolfinx.default_scalar_type(50)
interstitial_fluid_viscosity = dolfinx.default_scalar_type(1e-3)
#
# Permeability of the interstitium [m²]
# permeability      = dolfinx.default_scalar_type(1e-9*(1-0.65)**2) # 0.65 to equal t if permea_eps_square == False, set k_tilde = kf*(1-eps_eq^s)^2
permeability      = dolfinx.default_scalar_type(3.9e-14) 
permea_eps_square = False
#
# Reference/equilibrium solid volume fraction [-] 
equilibrium_fraction = dolfinx.default_scalar_type(0.68)
#
#
# Bulk compressibility modulus of the solid/scaffold phase [Pa]
bulk_modulus_solid = dolfinx.default_scalar_type(10.e3)
#
# Interfacial tension of the solid-fluid interface [Pa.m]
if interfaces:
	interfacial_tension = dolfinx.default_scalar_type(2e-3)
else:
	interfacial_tension = dolfinx.default_scalar_type(0.0)


#
# Radius of the solid (cell/particle) phase [m], sets the reference curvature
particle_radius = dolfinx.default_scalar_type(5.e-6)
#
# Reference curvature J^sf_0, computed from the particle radius [1/m]
curvature_reference = 2./particle_radius
#
# Reference specific interfacial density epsilon^sf_0 [1/m], computed from
# the equilibrium solid fraction and the reference curvature
interfacial_density_0 = (3./2.)*curvature_reference*equilibrium_fraction
#
# Sensitivity of the curvature J^sf to the solid fraction / to the
# interfacial density itself [-]
curvature_solid_coefficient       = dolfinx.default_scalar_type(-5)
curvature_interfacial_coefficient = dolfinx.default_scalar_type(0.)
#
# Second (bulk/dilatational) viscosity of the solid/scaffold phase [Pa.s],
# following the Stokes hypothesis lambda_s = -(2/3)*mu_s, as in the reference
# formulation. 
second_viscosity_solid = -(2./3.)*scafflod_like_viscosity
#
# Linearised (constrained) modulus of the solid phase, following the new
# linear pressure law (replaces the previous Byrne-law-based estimate)
# (M = bulk_modulus_solid/equilibrium_fraction)
M = bulk_modulus_solid
if mpi4py.MPI.COMM_WORLD.rank == 0:
	print(f'linearized longitudinal Modulus = {M}')
	print(f'Note: the interfacial tension contributes an extra d(J_sf.gamma_sf)/d(eps_s) '
		  f'= {curvature_solid_coefficient*interfacial_tension:.2e} Pa to this modulus, '
		  f'which is negligible here and is therefore not folded into cv below.')
#
if permea_eps_square == False:
	cv = permeability*M/interstitial_fluid_viscosity
else:
	cv = (permeability*(1-equilibrium_fraction)**2)*M/interstitial_fluid_viscosity
#
# --------------------------------------- #
# 	   		 Boundary values			  #
# --------------------------------------- #
# Pressures [Pa]
p_right = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0.))
# p_left  = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(100.))
p_left  = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(200.))
# For Terzaghi
p_left_terzaghi = 200
#
# --------------------------------------- #
# 	   			Operators				  #
# --------------------------------------- #
# Coordinates, for axisymmetry
x = ufl.SpatialCoordinate(domain)
# radius + small shift to avoid division by 0
r = abs(x[1]) #+1e-10
# Normals
normal = ufl.FacetNormal(domain)
# Specify the desired quadrature degree
q_deg  = 4
# attribute the cell_tags to the integral element
dx     = ufl.Measure('dx', metadata={"quadrature_degree":q_deg}, subdomain_data=cell_tags, domain=domain)
# attribute the facet_tags to the surfacic integral element
ds     = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
#
# --------------------------------------- #
# 	  	 		FunctionSpace			  #
# --------------------------------------- #
# Mixed Space (R,R,R2,R) -> (pl,eps_s,u,eps_sf) : fluid pressure, solid
# fraction, solid velocity, and (new formulation) specific interfacial density
P2_v      = basix.ufl.element("P", domain.topology.cell_name(), degree=2, shape=(domain.topology.dim,))
P1_v      = basix.ufl.element("P", domain.topology.cell_name(), degree=1, shape=(domain.topology.dim,))
P1        = basix.ufl.element("P", domain.topology.cell_name(), degree=1)
#
# Function spaces
P2v_space     = dolfinx.fem.functionspace(domain, P2_v)
P1v_space     = dolfinx.fem.functionspace(domain, P1_v)
P1_space      = dolfinx.fem.functionspace(domain, P1)
# Mixed-space
MS            = dolfinx.fem.functionspace(mesh=domain, element=basix.ufl.mixed_element([P1,P1,P2_v,P1]))
#
# --------------------------------------- #
# 	   			Functions				  #
# --------------------------------------- #
# Define the solution at the current and previous time steps
solution          = dolfinx.fem.Function(MS)
previous_solution = dolfinx.fem.Function(MS)
#
# Export P2 -> P1
velocity          = dolfinx.fem.Function(P1v_space)
#
# Post processing
porosity         = dolfinx.fem.Function(P1_space)
cell_pressure    = dolfinx.fem.Function(P1_space)
#
# Mapping in the Mixed Space: FunctionSpace, Mapping_in_the_mixed_space = MS.sub(xx).collapse()
Pressure_,    Pressure_to_MS    = MS.sub(0).collapse()
Fraction_,    Fraction_to_MS    = MS.sub(1).collapse()
Velocity_,    Velocity_to_MS    = MS.sub(2).collapse()
Interfacial_, Interfacial_to_MS = MS.sub(3).collapse()
#
# Create the test functions
q_p, q_e, q_v, q_i = ufl.TestFunctions(MS)
#
# --------------------------------------- #
# 	   		   Expressions			      #
# --------------------------------------- #
porosity_expr      = dolfinx.fem.Expression(1-previous_solution.sub(1) , P1_space.element.interpolation_points())
cell_pressure_expr = dolfinx.fem.Expression(constitutive_laws.Linear_pressure_law(previous_solution.sub(1),equilibrium_fraction,bulk_modulus_solid,log=True) , P1_space.element.interpolation_points())
# --------------------------------------- #
# 	   		Initial conditions			  #
# --------------------------------------- #
#
# NB: the fluid pressure field is intentionally left at its default value
# (0.) rather than pre-set to p_left. With the
# two-phase loading protocol, presetting it to p_left would be inconsistent
# with both the drained BC (pl=0 at x=0) and the fact that the load is not
# yet applied during Phase 1 -- this mismatch was found to trigger a large,
# hard-to-resolve initial transient with coarse time steps.
solution.x.array[Fraction_to_MS]    = dolfinx.default_scalar_type(equilibrium_fraction)
solution.x.array[Interfacial_to_MS] = dolfinx.default_scalar_type(interfacial_density_0)
solution.x.scatter_forward()
previous_solution.x.array[Fraction_to_MS]    = dolfinx.default_scalar_type(equilibrium_fraction)
previous_solution.x.array[Interfacial_to_MS] = dolfinx.default_scalar_type(interfacial_density_0)
previous_solution.x.scatter_forward()
#
# --------------------------------------- #
# 	   		Boundary conditions			  #
# --------------------------------------- #
# The interfacial density field is a purely advected/
# reactive quantity (no diffusion term in its transport equation), exactly
# like the solid fraction: neither field requires a Dirichlet boundary
# condition.
no_slip = False
bcs    = []
fdim   = domain.topology.dim - 1
#
# bottom, right, top, left = 1, 2, 3, 4
#
# bottom
facets = facet_tags.find(1)
# (vy = 0, tx = 0, grad(pl).n = 0)
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
# right
facets = facet_tags.find(2)
# (vx = 0, vy = 0, pl = 0)
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(0), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(0)))
#
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
#
# top
facets = facet_tags.find(3)
# No-slip condition
if no_slip:
	dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(0), fdim, facets)
	bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(0)))
#
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
# left
facets = facet_tags.find(4)
# (tx = p0, ty = 0, grad(pl).n = 0)
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(0), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(0)))
#
# --------------------------------------- #
# 	   			Weak Form				  #
# --------------------------------------- #
#
fluid_pressure,   cell_fraction,   cell_velocity,   interfacial_density   = ufl.split(solution)
fluid_pressure_n, cell_fraction_n, cell_velocity_n, interfacial_density_n = ufl.split(previous_solution)
#
# Interfacial curvature J^sf (new formulation)
curvature = constitutive_laws.Interfacial_curvature(
	cell_fraction, interfacial_density, equilibrium_fraction, interfacial_density_0,
	curvature_reference, curvature_solid_coefficient, curvature_interfacial_coefficient
	)
#
# Weak form
# mass epsilon^s (solid fraction transport)
F = (
	(1/dt)*(cell_fraction-cell_fraction_n)*q_e*r*dx
	+ functions.div_cyl_x(cell_fraction*cell_velocity,x)*q_e*r*dx
	)
# Mixture mass conservation (new formulation): replaces the previous direct
# fluid-fraction balance ((1-eps_s) time-derivative + flux) by the summed
# solid+fluid mass balance div(u) + Darcy term, consistently with the
# reference formulation now that epsilon^sf carries its own transport
# equation and provides the extra equation needed to close
# the system with 4 unknowns.
F+= q_p*functions.div_cyl_x(cell_velocity,x)*r*dx
if permea_eps_square == False:
	F+= ((permeability)/interstitial_fluid_viscosity)*ufl.dot(ufl.grad(fluid_pressure), ufl.grad(q_p))*r*dx
else:
	F+= ((permeability*(1-cell_fraction)**2)/interstitial_fluid_viscosity)*ufl.dot(ufl.grad(fluid_pressure), ufl.grad(q_p))*r*dx
#
# Interfacial density transport (new formulation) 
F+= (
	(1/dt)*(interfacial_density-interfacial_density_n)*q_i*r*dx
	+ (curvature*cell_fraction + (1/3)*interfacial_density)*functions.div_cyl_x(cell_velocity,x)*q_i*r*dx
	+ ufl.dot(ufl.grad(interfacial_density), cell_velocity)*q_i*r*dx
	)
#
# test velocity (momentum balance), common to both loading phases
F_momentum_common = (
	- fluid_pressure*functions.div_cyl_x(q_v,x)*r*dx
	- (constitutive_laws.Linear_pressure_law(cell_fraction,equilibrium_fraction,bulk_modulus_solid,log=True))*functions.div_cyl_x(q_v,x)*r*dx
	- interfacial_tension*curvature*functions.div_cyl_x(q_v,x)*r*dx
	+ scafflod_like_viscosity*ufl.inner(functions.grad_cyl_x(cell_velocity,x), functions.grad_cyl_x(q_v,x))*r*dx
	+ scafflod_like_viscosity*ufl.inner(functions.grad_cyl_x(cell_velocity,x).T, functions.grad_cyl_x(q_v,x))*r*dx
	+ second_viscosity_solid*functions.div_cyl_x(cell_velocity,x)*functions.div_cyl_x(q_v,x)*r*dx
	)

# Neumann traction load p_left, active only during the loaded phase (TPHASE2):
# two-phase loading protocol introduced with the new formulation.
F_traction = p_left*ufl.dot(normal, q_v)*r*ds(4)
#
# Phase 1 (equilibration, no external load) and Phase 2 (loaded) weak forms
F_phase1 = F + F_momentum_common
F_phase2 = F + F_momentum_common + F_traction
#
# --------------------------------------- #
# 	   				Solver			 	  #
# --------------------------------------- #
# Non linear problem definition, built independently for each loading phase
def build_problem(F_form):
	"""Builds a NonlinearProblem and its Newton solver for a given weak form."""
	dsolution_ = ufl.TrialFunction(MS)
	J_         = ufl.derivative(F_form, solution, dsolution_)
	Problem_   = NonlinearProblem(F_form, solution, bcs = bcs, J = J_)
	# NB: rtol relaxed from 1e-12 (the value inherited from the original
	# 3-field/small-load Terzaghi test) to 1e-6, matching the reference's own
	# tolerance. The new interfacial density field eps_sf has values of order
	# Jsf0*eps_s0 ~ 1/particle_radius (~1e5-1e6 here), five to six orders of
	# magnitude above the other unknowns (eps_s ~ O(1), pl ~ O(1-200),
	# v ~ O(1e-8)); combined with a monolithic block solve, rtol=1e-12 is
	# effectively unreachable and was exhausting max_it, which the try/except
	# around solver.solve(...) then reports as "Solver failed".
	solver_    = functions.set_non_linear_solver_parameters(domain, Problem_, 1e-8, 1e-10, "incremental", 50, log_newton=False)
	return Problem_, solver_
#
# --------------------------------------- #
# 	   	  Solving & Post-treatment		  #
# --------------------------------------- #
#
# vtk outputs_fluid_terzaghi
pressure_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_fluid_terzaghi/VTK/pressure.pvd", "w")
pressure_vtk.write_mesh(domain)
fraction_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_fluid_terzaghi/VTK/fraction.pvd", "w")
fraction_vtk.write_mesh(domain)
velocity_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_fluid_terzaghi/VTK/velocity.pvd", "w")
velocity_vtk.write_mesh(domain)
porosity_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_fluid_terzaghi/VTK/porosity.pvd", "w")
porosity_vtk.write_mesh(domain)
cell_pressure_vtk  = dolfinx.io.VTKFile(domain.comm, "outputs_fluid_terzaghi/VTK/cell_pressure.pvd", "w")
cell_pressure_vtk.write_mesh(domain)
interfacial_vtk    = dolfinx.io.VTKFile(domain.comm, "outputs_fluid_terzaghi/VTK/interfacial_density.pvd", "w")
interfacial_vtk.write_mesh(domain)
# ------------------------------------------------------------------------ #
#
# Create output lists in time and space for the IF pressure
pressure_x_L          = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
pressure_x_L_over_2   = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
porosity_all_L        = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
porosity_all_L_over_2 = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
fraction_all          = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
velocity_0            = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
velocity_L_over_2     = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
# continuous tracking of the interfacial density field
interfacial_density_all_L          = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
interfacial_density_all_L_over_2   = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)

pressure_x_0                      = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
fraction_all_L_over_2             = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
fraction_all_0                    = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
velocity_L                        = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
interfacial_density_all_0         = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
#
pressure_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressure_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressure_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
velocity_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
velocity_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
velocity_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
# spatial profile of the interfacial density field
interfacial_density_space0 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
interfacial_density_space1 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
interfacial_density_space2 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
#
# Error (only meaningful once the load is applied, i.e. during Phase 2)
L2_p = numpy.zeros(num_steps_phase2, dtype=dolfinx.default_scalar_type)
L2_v = numpy.zeros(num_steps_phase2, dtype=dolfinx.default_scalar_type)
# in space
Signed_p0 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
Signed_v0 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
Signed_p1 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
Signed_v1 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
Signed_p2 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
Signed_v2 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
#
# ------------------------------------------------------------------------ #
#
# initialise time
t = 0
# Export initial solution at t=0s
__pressure, __fraction, __v, __interfacial  = previous_solution.split()
__pressure.name              = "Fluid Pressure"
__fraction.name              = "Cell Fraction"
__interfacial.name           = "Interfacial Density"
velocity.interpolate(previous_solution.sub(2))
velocity.x.scatter_forward()
velocity.name                = "Cell Velocity"
porosity.interpolate(porosity_expr)
porosity.x.scatter_forward()
porosity.name                = "Porosity"
cell_pressure.interpolate(cell_pressure_expr)
cell_pressure.x.scatter_forward()
cell_pressure.name                = "cell pressure"
#
# Initial step in lists
evaluate_point(domain, __pressure, cells_x_L, points[0], pressure_x_L, 0)
evaluate_point(domain, __pressure, cells_x_L_over_2, points[1], pressure_x_L_over_2, 0)
evaluate_point(domain, __pressure, cells_x_0, points[2], pressure_x_0, 0)
evaluate_point(domain, porosity, cells_x_L, points[0], porosity_all_L, 0)
evaluate_point(domain, porosity, cells_x_L_over_2, points[1], porosity_all_L_over_2, 0)
# NB: fraction_all is evaluated at cells_x_L (PC) here to match the in-loop
# evaluation below (was previously evaluated at cells_x_L_over_2 at t=0 only
# -- a bug that made the very first sample of "fraction_all" come from a
# different physical point, x=L/2, than every subsequent sample, x=L).
evaluate_point(domain, __fraction, cells_x_L, points[0], fraction_all, 0)
evaluate_point(domain, __fraction, cells_x_L_over_2, points[1], fraction_all_L_over_2, 0)
evaluate_point(domain, __fraction, cells_x_0, points[2], fraction_all_0, 0)
evaluate_point(domain, __interfacial, cells_x_L, points[0], interfacial_density_all_L, 0)
evaluate_point(domain, __interfacial, cells_x_L_over_2, points[1], interfacial_density_all_L_over_2, 0)
evaluate_point(domain, __interfacial, cells_x_0, points[2], interfacial_density_all_0, 0)
# Export in the vtk files
pressure_vtk.write_function(__pressure.collapse(),t)
fraction_vtk.write_function(__fraction.collapse(),t)
velocity_vtk.write_function(velocity,t)
porosity_vtk.write_function(porosity,t)
cell_pressure_vtk.write_function(cell_pressure,t)
interfacial_vtk.write_function(__interfacial.collapse(),t)
#
# --------------------------------------- #
# 	  	  Time stepping (2 phases)		  #
# --------------------------------------- #
# Phase 1: equilibration, no external load (skipped entirely if TPHASE1 = 0.)
# Phase 2: loaded (consolidation) phase, compared against the Terzaghi
#          analytical solution.
# A single loop body is reused for both phases (DRY), with the L2 error and
# the fine time/space sampling used for the Terzaghi comparison only computed
# during Phase 2, since no load (hence no analytical reference) exists before.
# ------------------------------------------------------------------------- #
phases = []
if num_steps_phase1 > 0:
	phases.append({"name": "Phase 1 (equilibration)", "F": F_phase1, "n_steps": num_steps_phase1, "is_loaded": False})
phases.append({"name": "Phase 2 (loaded)", "F": F_phase2, "n_steps": num_steps_phase2, "is_loaded": True})
#
global_iteration = 0
# Sampling times for the detailed spatial profiles (n0, n1, n2 steps into
# Phase 2). Left as None if the solver fails before a given step is reached,
# so the post-processing block below can detect and skip it instead of
# crashing on an undefined variable.
t0 = t1 = t2 = None
for phase in phases:
	Problem, solver = build_problem(phase["F"])
	if mpi4py.MPI.COMM_WORLD.rank == 0:
		print(f"--- Starting {phase['name']}: {phase['n_steps']} steps ---")
	for local_iteration in range(phase["n_steps"]):
		# increment time
		t+=dt
		global_iteration += 1
		#
		if mpi4py.MPI.COMM_WORLD.rank == 0:
			print('Step: ',global_iteration,'/',num_steps, 'time:', t, ' s')
		# Solve
		try:
			num_its, converged = solver.solve(solution)
		except:
			if mpi4py.MPI.COMM_WORLD.rank == 0:
				print("*************")
				print("Solver failed")
				print("*************")
			break
		# Ensure pushing the solution to all processes
		solution.x.scatter_forward()
		# Update previous solution
		previous_solution.x.array[:] = solution.x.array[:]
		previous_solution.x.scatter_forward()
		# Export
		__pressure, __fraction, __v, __interfacial  = previous_solution.split()
		__pressure.name              = "Fluid Pressure"
		__fraction.name              = "Cell Fraction"
		__interfacial.name           = "Interfacial Density"
		#
		velocity.interpolate(__v)
		velocity.x.scatter_forward()
		velocity.name                = "Cell Velocity"
		#
		porosity.interpolate(porosity_expr)
		porosity.x.scatter_forward()
		porosity.name                = "Porosity"
		cell_pressure.interpolate(cell_pressure_expr)
		cell_pressure.x.scatter_forward()
		cell_pressure.name                = "cell pressure"
		if global_iteration%1 == 0:
			# Export in the vtk files
			pressure_vtk.write_function(__pressure.collapse(),t)
			fraction_vtk.write_function(__fraction.collapse(),t)
			velocity_vtk.write_function(velocity,t)
			porosity_vtk.write_function(porosity,t)
			cell_pressure_vtk.write_function(cell_pressure,t)
			interfacial_vtk.write_function(__interfacial.collapse(),t)
		# Evaluate the functions
		# in time (both phases)
		evaluate_point(domain, __pressure, cells_x_L, points[0], pressure_x_L, global_iteration)
		evaluate_point(domain, __pressure, cells_x_L_over_2, points[1], pressure_x_L_over_2, global_iteration)
		evaluate_point(domain, __pressure, cells_x_0, points[2], pressure_x_0, global_iteration)
		evaluate_point(domain, porosity, cells_x_L, points[0], porosity_all_L, global_iteration)
		evaluate_point(domain, porosity, cells_x_L_over_2, points[1], porosity_all_L_over_2, global_iteration)
		evaluate_point(domain, __fraction, cells_x_L, points[0], fraction_all, global_iteration)
		evaluate_point(domain, __fraction, cells_x_L_over_2, points[1], fraction_all_L_over_2, global_iteration)
		evaluate_point(domain, __fraction, cells_x_0, points[2], fraction_all_0, global_iteration)
		evaluate_point(domain, velocity.sub(0), cells_x_0, points[2], velocity_0, global_iteration)
		evaluate_point(domain, velocity.sub(0), cells_x_L_over_2, points[1], velocity_L_over_2, global_iteration)
		evaluate_point(domain, velocity.sub(0), cells_x_L, points[0], velocity_L, global_iteration)
		evaluate_point(domain, __interfacial, cells_x_L, points[0], interfacial_density_all_L, global_iteration)
		evaluate_point(domain, __interfacial, cells_x_L_over_2, points[1], interfacial_density_all_L_over_2, global_iteration)
		evaluate_point(domain, __interfacial, cells_x_0, points[2], interfacial_density_all_0, global_iteration)
		#
		# Terzaghi comparison: only meaningful once the load is applied (Phase 2)
		if phase["is_loaded"]:
			# Compute L2 norm for pressure
			error_L2p     = L2_error_p(domain,P1,__pressure)
			L2_p[local_iteration] = error_L2p
			error_L2v     = L2_error_v(domain,P1,velocity)
			L2_v[local_iteration] = error_L2v
			if mpi4py.MPI.COMM_WORLD.rank == 0:
				print(f"L2-error p {error_L2p:.2e} [Pa]")
				print(f"L2-error v {error_L2v:.2e} [m/s]")
			# in space, counted from the start of the loaded phase
			if local_iteration == n0:
				for ii in range(num_points):
					evaluate_point(domain, __pressure, colliding_cells.links(ii+2), points[ii+2], pressure_space0, ii)
					evaluate_point(domain, velocity.sub(0), colliding_cells.links(ii+2), points[ii+2], velocity_space0, ii)
					evaluate_point(domain, __interfacial, colliding_cells.links(ii+2), points[ii+2], interfacial_density_space0, ii)
				t0 = t
			elif local_iteration == n1:
				for ii in range(num_points):
					evaluate_point(domain, __pressure, colliding_cells.links(ii+2), points[ii+2], pressure_space1, ii)
					evaluate_point(domain, velocity.sub(0), colliding_cells.links(ii+2), points[ii+2], velocity_space1, ii)
					evaluate_point(domain, __interfacial, colliding_cells.links(ii+2), points[ii+2], interfacial_density_space1, ii)
				t1 = t
			elif local_iteration == n2:
				for ii in range(num_points):
					evaluate_point(domain, __pressure, colliding_cells.links(ii+2), points[ii+2], pressure_space2, ii)
					evaluate_point(domain, velocity.sub(0), colliding_cells.links(ii+2), points[ii+2], velocity_space2, ii)
					evaluate_point(domain, __interfacial, colliding_cells.links(ii+2), points[ii+2], interfacial_density_space2, ii)
				t2 = t
		#
#
# --------------------------------------- #
# 	   	  Post-processing Terzaghi		  #
# --------------------------------------- #


# Discriminate the interfaces=True/False configurations in the CSV output
# filenames: both use the same permeability value in the filename, so
# without this an interfaces=True run would silently overwrite the CSVs
# from an interfaces=False run (and vice versa). The interfacial tension
# (interfacial_tension, gamma_sf) is also embedded directly in the
# filenames (next to permeability, defined further below), so that
# sweeping gamma_sf (e.g. 0, 0.002, 0.004 N/m) for parametric comparison
# plots produces distinct, non-overwriting CSVs too.
interfaces_suffix = "_"+str(interfacial_tension)+"_interfaces" if interfaces else ""

# print some error statistics
if mpi4py.MPI.COMM_WORLD.rank == 0:
	print(f"L2 error p, min {numpy.min(L2_p):.2e} Pa, mean {numpy.mean(L2_p):.2e} Pa, max {numpy.max(L2_p):.2e} Pa, std {numpy.std(L2_p):.2e} Pa")
	print(f"L2 error v, min {numpy.min(L2_v):.2e} m/s, mean {numpy.mean(L2_v):.2e} m/s, max {numpy.max(L2_v):.2e} m/s, std {numpy.std(L2_v):.2e} m/s")
	print("Normalised to max")
	print(f"L2 error p, min {numpy.min([100*x/numpy.max(pressure_x_L) for x in L2_p]):.2e} %, mean {numpy.mean([100*x/numpy.max(pressure_x_L) for x in L2_p]):.2e} %, max {numpy.max([100*x/numpy.max(pressure_x_L) for x in L2_p]):.2e} %, std {numpy.std([100*x/numpy.max(pressure_x_L) for x in L2_p]):.2e} %")
	print(f"L2 error v, min {numpy.min([100*x/numpy.max(velocity_0) for x in L2_v]):.2e} %, mean {numpy.mean([100*x/numpy.max(velocity_0) for x in L2_v]):.2e} %, max {numpy.max([100*x/numpy.max(velocity_0) for x in L2_v]):.2e} %, std {numpy.std([100*x/numpy.max(velocity_0) for x in L2_v]):.2e} %")
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if mpi4py.MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
#
if mpi4py.MPI.COMM_WORLD.rank == 0:
	###################################################
	################ Analytical solutions #############
	###################################################
	#
	# Elapsed time is taken relative to TPHASE1, the start of the loaded phase
	# (two-phase loading protocol, new formulation). tstart = 0. reproduces the
	# original single-phase benchmark exactly.
	tstart = TPHASE1
	y=0
	t=numpy.linspace(0,t_fin,num_steps+1)
	pressure4 = numpy.zeros(num_steps+1)
	kmax=1e3
	for i in range(num_steps+1):
		# No consolidation has started yet while t <= TPHASE1 (equilibration phase)
		if t[i] <= tstart:
			pressure4[i] = 0.0
		else:
			pressure4[i] = functions.terzaghi(p_left_terzaghi,length_cylinder,cv,y,t[i],tstart,int(kmax))
	#
	y=length_cylinder/2
	t=numpy.linspace(0,t_fin,num_steps+1)
	pressure5 = numpy.zeros(num_steps+1)
	for i in range(num_steps+1):
		if t[i] <= tstart:
			pressure5[i] = 0.0
		else:
			pressure5[i] = functions.terzaghi(p_left_terzaghi,length_cylinder,cv,y,t[i],tstart,int(kmax))
	#
	# If the solver failed before a given sampling step was reached, t0/t1/t2
	# stay None: fall back to NaN instead of crashing, so the run still
	# produces its (partial) CSV outputs for inspection.
	if t0 is None:
		if mpi4py.MPI.COMM_WORLD.rank == 0:
			print(f"Warning: step n0={n0} (Phase 2) was never reached, pressure_space0/velocity_space0 will be NaN")
		pressure0 = numpy.full(num_points, numpy.nan)
		velocity0 = numpy.full(num_points, numpy.nan)
	else:
		pressure0 = numpy.zeros(num_points)
		for i in range(num_points):
			pressure0[i] = functions.terzaghi(p_left_terzaghi,length_cylinder,cv,x_check[i],t0,tstart,int(kmax))
		velocity0 = numpy.zeros(num_points)
		for i in range(num_points):
			velocity0[i] = functions.terzaghi_v(p_left_terzaghi,length_cylinder,cv,x_check[i],t0,tstart,int(kmax),M)
	#
	if t1 is None:
		if mpi4py.MPI.COMM_WORLD.rank == 0:
			print(f"Warning: step n1={n1} (Phase 2) was never reached, pressure_space1/velocity_space1 will be NaN")
		pressure1 = numpy.full(num_points, numpy.nan)
		velocity1 = numpy.full(num_points, numpy.nan)
	else:
		pressure1 = numpy.zeros(num_points)
		for i in range(num_points):
			pressure1[i] = functions.terzaghi(p_left_terzaghi,length_cylinder,cv,x_check[i],t1,tstart,int(kmax))
		velocity1 = numpy.zeros(num_points)
		for i in range(num_points):
			velocity1[i] = functions.terzaghi_v(p_left_terzaghi,length_cylinder,cv,x_check[i],t1,tstart,int(kmax),M)
	#
	if t2 is None:
		if mpi4py.MPI.COMM_WORLD.rank == 0:
			print(f"Warning: step n2={n2} (Phase 2) was never reached, pressure_space2/velocity_space2 will be NaN")
		pressure2 = numpy.full(num_points, numpy.nan)
		velocity2 = numpy.full(num_points, numpy.nan)
	else:
		pressure2 = numpy.zeros(num_points)
		for i in range(num_points):
			pressure2[i] = functions.terzaghi(p_left_terzaghi,length_cylinder,cv,x_check[i],t2,tstart,int(kmax))
		velocity2 = numpy.zeros(num_points)
		for i in range(num_points):
			velocity2[i] = functions.terzaghi_v(p_left_terzaghi,length_cylinder,cv,x_check[i],t2,tstart,int(kmax),M)
	#
	Signed_p0 = [(pressure_space0[index]-pressure0[index]) for index in range(num_points)]
	Signed_p1 = [(pressure_space1[index]-pressure1[index]) for index in range(num_points)]
	Signed_p2 = [(pressure_space2[index]-pressure2[index]) for index in range(num_points)]
	Signed_v0 = [(velocity_space0[index]-velocity0[index]) for index in range(num_points)]
	Signed_v1 = [(velocity_space1[index]-velocity1[index]) for index in range(num_points)]
	Signed_v2 = [(velocity_space2[index]-velocity2[index]) for index in range(num_points)]
	#
	# L2_p/L2_v are only meaningful during Phase 2 (loaded phase) and are
	# therefore sized num_steps_phase2, unlike every other time-series column
	# here (t, pressure_x_L, ...) which is sized num_steps+1. export_to_csv
	# writes each array as one raw CSV row (of possibly different length --
	# csv.writer raises no error for that), but transpose_csv later transposes
	# with zip(*data), which silently stops at the SHORTEST row. Mixing a
	# 6000-long row with 12001-long rows therefore truncated every column of
	# the transposed (pgfplots-facing) CSV to 6000 rows (30h) regardless of
	# gamma -- with no error anywhere in the pipeline. Fix: pad L2_p/L2_v (and
	# their normalised counterparts) with NaN over Phase 1 so all exported
	# columns share the same length; pgfplots simply skips NaN points.
	L2_p_full = numpy.full(num_steps+1, numpy.nan, dtype=dolfinx.default_scalar_type)
	L2_v_full = numpy.full(num_steps+1, numpy.nan, dtype=dolfinx.default_scalar_type)
	phase2_start = num_steps_phase1 + 1
	L2_p_full[phase2_start:phase2_start+num_steps_phase2] = L2_p
	L2_v_full[phase2_start:phase2_start+num_steps_phase2] = L2_v
	#
	# CSV export: the existing
	# column names/order are kept unchanged; the new formulation's interfacial
	# density data is appended as EXTRA columns at the end of each file.
	functions.export_to_csv(
		[x_check,pressure_space0,pressure_space1,pressure_space2,velocity_space0,velocity_space1,velocity_space2,Signed_p0,Signed_v0,Signed_p1,Signed_v1,Signed_p2,Signed_v2,interfacial_density_space0,interfacial_density_space1,interfacial_density_space2],
		"outputs_fluid_terzaghi/f_Results_x_"+str(permeability)+interfaces_suffix+".csv",
		["x","pressure_space0","pressure_space1","pressure_space2","velocity_space0","velocity_space1","velocity_space2","Signed_p0","Signed_v0","Signed_p1","Signed_v1","Signed_p2","Signed_v2","interfacial_density_space0","interfacial_density_space1","interfacial_density_space2"]
		)
	functions.export_to_csv(
		[t,porosity_all_L,porosity_all_L_over_2,fraction_all,velocity_0,velocity_L_over_2,pressure_x_L,pressure_x_L_over_2,L2_p_full,L2_v_full, [x/numpy.max(pressure_x_L) for x in L2_p_full], [x/numpy.max(velocity_0) for x in L2_v_full],interfacial_density_all_L,interfacial_density_all_L_over_2,
		 pressure_x_0,fraction_all_L_over_2,fraction_all_0,velocity_L,interfacial_density_all_0],
		"outputs_fluid_terzaghi/f_Results_t_"+str(permeability)+interfaces_suffix+".csv",
		["t","porosity_all_L","porosity_all_L_over_2","fraction_all","velocity_0","velocity_L_over_2","pressure_x_L","pressure_x_L_over_2","L2_p","L2_v","L2_p_normmax","L2_v_normmax","interfacial_density_all_L","interfacial_density_all_L_over_2",
		 "pressure_x_0","fraction_all_L_over_2","fraction_all_0","velocity_L","interfacial_density_all_0"]
		)
	functions.export_to_csv(
		[x_check,pressure0,pressure1,pressure2,velocity0,velocity1,velocity2],
		"outputs_fluid_terzaghi/f_Terzaghi_x_"+str(permeability)+interfaces_suffix+".csv",
		["x","pressure0","pressure1","pressure2","velocity0","velocity1","velocity2"]
		)
	functions.export_to_csv(
		[t,pressure4,pressure5],
		"outputs_fluid_terzaghi/f_Terzaghi_t_"+str(permeability)+interfaces_suffix+".csv",
		["t","pressure4","pressure5"]
		)
	#
	def transpose_csv(input_file,output_file):
		import csv
		with open(input_file, newline='') as f:
		    reader = csv.reader(f)
		    rows = list(reader)

		# Séparer le header et les données
		header = rows[0]       # première ligne
		data = rows[1:]        # toutes les autres lignes

		# Transposer seulement les données
		transposed_data = list(map(list, zip(*data)))

		# Réécrire le CSV avec le header inchangé
		with open(output_file, 'w', newline='') as f:
		    writer = csv.writer(f)
		    writer.writerow(header)           # écrire le header original
		    writer.writerows(transposed_data) # écrire les données transposées
		print(f"Transposition terminée pour latex: {output_file}")

	input_file = "outputs_fluid_terzaghi/f_Results_t_"+str(permeability)+interfaces_suffix+".csv"
	output_file = "outputs_fluid_terzaghi/ft_Results_t_"+str(permeability)+interfaces_suffix+".csv"
	transpose_csv(input_file,output_file)

	input_file = "outputs_fluid_terzaghi/f_Results_x_"+str(permeability)+interfaces_suffix+".csv"
	output_file = "outputs_fluid_terzaghi/ft_Results_x_"+str(permeability)+interfaces_suffix+".csv"
	transpose_csv(input_file,output_file)

	input_file = "outputs_fluid_terzaghi/f_Terzaghi_x_"+str(permeability)+interfaces_suffix+".csv"
	output_file = "outputs_fluid_terzaghi/ft_Terzaghi_x_"+str(permeability)+interfaces_suffix+".csv"
	transpose_csv(input_file,output_file)

	input_file = "outputs_fluid_terzaghi/f_Terzaghi_t_"+str(permeability)+interfaces_suffix+".csv"
	output_file = "outputs_fluid_terzaghi/ft_Terzaghi_t_"+str(permeability)+interfaces_suffix+".csv"
	transpose_csv(input_file,output_file)


#EoF
