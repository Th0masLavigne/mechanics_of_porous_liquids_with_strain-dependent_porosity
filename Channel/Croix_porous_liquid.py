# Date: 09/09/2025
# Author: Thomas Lavigne
# Reference: Giuseppe Sciumè
# Laboratory: I2M
# 
# 
# --------------------------------------- #
# 			Import libraries			  #
# --------------------------------------- #
import sys
import os
import shutil
import dolfinx
import mpi4py
import ufl
import basix
import numpy
from dolfinx.fem.petsc import NonlinearProblem
import time
import argparse
# 
if mpi4py.MPI.COMM_WORLD == 0:
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
sys.path.append('./service_files_channel')
# 
import geometry_porous_liquid
import constitutive_laws
import functions
# 
# --------------------------------------- #
# 	   	Command-line parameters		  #
# --------------------------------------- #
# Defaults reproduce the simulation reported in the manuscript
# (dt = 0.01 s, Gmsh size_max = 1 mm -> mean element size ~1.02 mm).
# They are overridden by run_convergence_study.sh for the spatial and
# temporal convergence study (Supplementary Material).
parser = argparse.ArgumentParser(description="Porous liquid flow in a bifurcated (T-shaped) channel")
parser.add_argument("--dt",        type=float, default=1e-2, help="time step [s]")
parser.add_argument("--t_fin",     type=float, default=5.,   help="final time [s]")
parser.add_argument("--size_max",  type=float, default=1e-3, help="Gmsh Mesh.MeshSizeMax [m]")
parser.add_argument("--size_min",  type=float, default=None, help="Gmsh Mesh.MeshSizeMin [m] (default: size_max/10)")
parser.add_argument("--tag",       type=str,   default="",   help="suffix of the output folder / mesh file")
parser.add_argument("--no_fields", action="store_true",      help="skip the XDMF/VTK export of the fields (convergence runs)")
args = parser.parse_args()
size_max = args.size_max
size_min = args.size_min if args.size_min is not None else args.size_max/10
# 
# path to the ouptut
folder = f"./outputs_croix{args.tag}/"
# legacy csv files are written in the working directory for the reference run,
# and in the output folder for tagged (convergence) runs to avoid overwriting
csv_dir = "." if args.tag == "" else folder
# 
# --------------------------------------- #
# 	   			Clean workspace	 		  #
# --------------------------------------- #
# 
# remove old output if exists
if mpi4py.MPI.COMM_WORLD.rank == 0:
	if os.path.exists(folder) and os.path.isdir(folder):
		shutil.rmtree(folder)
		print(f"Removed folder: {folder}")
	else:
		print(f"Folder does not exist: {folder}")
	os.makedirs(folder, exist_ok=True)
mpi4py.MPI.COMM_WORLD.Barrier()
# 
# 
# --------------------------------------- #
# 	   			 Log level				  #
# --------------------------------------- #
# 
log      = False
#
log_mesh = False
# 
create_new_mesh = True
# 
def evaluate_point(mesh, function, contributing_cells, point, output_list, index):
	"""
	Suitable Evaluations functions for Parallel computation
	Inputs: mesh, function to evaluate, contributing cells to the point, point, output list to store the value, index in the list
	outputs_fluid_terzaghi: the evaluated function value is added at the index location in output list
	"""
	function_eval = None
	if len(contributing_cells) > 0:
		function_eval = function.eval(point, contributing_cells[:1])
	function_eval = mesh.comm.gather(function_eval, root=0)
	# Choose first pressure that is found from the different processors
	if mpi4py.MPI.COMM_WORLD.rank == 0:
		for element in function_eval:
			if element is not None:
				output_list[index]=element[0]
				break
	return None
# 
# --------------------------------------- #
# 	   		 Time parameters			  #
# --------------------------------------- #
#
#
t_fin = args.t_fin
# Time step [s]. Original value (0.005h).
dt    = args.dt
#
num_steps = int(round(t_fin/dt))
#
# time steps to evaluate the pressure/velocity profiles in space 
n0, n1, n2, n3, n4 = 10, 70, 115, 300, 450
# keep the same physical snapshot times if dt differs from the reference 0.01 s
n0, n1, n2, n3, n4 = [int(round(n*1e-2/dt)) for n in (n0, n1, n2, n3, n4)]
# --------------------------------------- #
# 	   	Geometrical parameters			  #
# --------------------------------------- #
# 
L1,H1,l2,H2 = 1e-1, 2e-2,1e-2,5e-2
# 
# --------------------------------------- #
# 	   		Generate Mesh				  #
# --------------------------------------- #
# Create gmsh tagged mesh
gmsh_name = f"geometry/2D_T_mesh{args.tag}.msh"
# 
if (os.path.exists(gmsh_name)) and (create_new_mesh == False):
	pass
else:
	if mpi4py.MPI.COMM_WORLD.rank == 0:
		# Reverted to the original mesh resolution (1e-4/1e-3), matching the
		# "2nd experiment" baseline being reproduced here.
		geometry_porous_liquid.T_mesh(L1,H1,l2,H2, size_min=size_min, size_max=size_max, log=True, optimise=True, netgen=True, filename=gmsh_name)
# 
# Make sure the mesh file is written before the other ranks read it
mpi4py.MPI.COMM_WORLD.Barrier()
# Read the mesh and tags
gmsh_model_rank = 0
gdim            = 2
# 
domain, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(gmsh_name, mpi4py.MPI.COMM_WORLD, gmsh_model_rank, gdim=gdim)
# 
# # 
if log_mesh:
	xdmftag = dolfinx.io.XDMFFile(mpi4py.MPI.COMM_WORLD, folder+'debug_tags_mesh.xdmf', "w")
	xdmftag.write_mesh(domain)
	xdmftag.write_meshtags(facet_tags, domain.geometry)
	xdmftag.write_meshtags(cell_tags,  domain.geometry)
	xdmftag.close()
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
num_points = 15
# Physical points we want an evaluation in
x_check          = numpy.linspace(0,L1,num_points)
y_check 			 = numpy.linspace(0,H1,num_points)
points_for_space_y_L = numpy.zeros((num_points,3))
points_for_space_y_R = numpy.zeros((num_points,3))
points_for_space_x   = numpy.zeros((num_points,3))
for ii in range(num_points):
	points_for_space_x[ii,1] = 0.
	points_for_space_x[ii,0] = x_check[ii]
	points_for_space_y_L[ii,1] = y_check[ii]
	points_for_space_y_L[ii,0] = L1/10
	points_for_space_y_R[ii,1] = y_check[ii]
	points_for_space_y_R[ii,0] = 9*L1/10
# Create the bounding box tree and identify contributing cells
tree             = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
points           = numpy.concatenate((points_for_space_y_L,points_for_space_y_R,points_for_space_x))
cell_candidates  = dolfinx.geometry.compute_collisions_points(tree, points)
colliding_cells  = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points)
#
# diagnostic point, kept out of the `points`/`colliding_cells`
branch_point            = numpy.array([[L1/2, H1+H2/2, 0.]])
branch_cell_candidates  = dolfinx.geometry.compute_collisions_points(tree, branch_point)
branch_colliding_cells  = dolfinx.geometry.compute_colliding_cells(domain, branch_cell_candidates, branch_point)
#
# Convergence-study probes: dense vx(y) profiles at x = 0.1 L and x = 0.9 L
# (same sections as Figure 12) + the branch mid-point (infiltration velocity).
num_conv    = 41
y_conv      = numpy.linspace(0, H1, num_conv)
conv_points = numpy.zeros((2*num_conv+1, 3))
conv_points[:num_conv, 0]           = L1/10
conv_points[:num_conv, 1]           = y_conv
conv_points[num_conv:2*num_conv, 0] = 9*L1/10
conv_points[num_conv:2*num_conv, 1] = y_conv
conv_points[2*num_conv]             = branch_point[0]
conv_cell_candidates = dolfinx.geometry.compute_collisions_points(tree, conv_points)
conv_colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, conv_cell_candidates, conv_points)
#
def evaluate_points(mesh, function, colliding_cells, pts):
	"""
	Evaluate a (vector) function at all points in a single call (MPI safe).
	Returns an array (num_points, value_size) on rank 0, None elsewhere.
	"""
	ids, cells = [], []
	for i in range(pts.shape[0]):
		links = colliding_cells.links(i)
		if len(links) > 0:
			ids.append(i)
			cells.append(links[0])
	vals = None
	if len(ids) > 0:
		vals = function.eval(pts[ids], numpy.asarray(cells, dtype=numpy.int32))
	gathered = mesh.comm.gather((ids, vals), root=0)
	if mesh.comm.rank != 0:
		return None
	out = None
	for g_ids, g_vals in gathered:
		if g_vals is None:
			continue
		g_vals = numpy.asarray(g_vals).reshape(len(g_ids), -1)
		if out is None:
			out = numpy.full((pts.shape[0], g_vals.shape[1]), numpy.nan)
		for k, i in enumerate(g_ids):
			if numpy.isnan(out[i, 0]):
				out[i] = g_vals[k]
	return out
#
# --------------------------------------- #
# 	   		Material parameters			  #
# --------------------------------------- #
# 
# Viscosities [Pa.s]
scafflod_like_viscosity      = dolfinx.default_scalar_type(50)
interstitial_fluid_viscosity = dolfinx.default_scalar_type(1e-3)
#
# Second (bulk/dilatational) viscosity of the solid/scaffold phase [Pa.s],
# following the Stokes hypothesis lambda_s = -(2/3)*mu_s, as introduced in
# the reference formulation porous_liquid_2026_all3.py and already applied
# to the corrected Consolidation model (Porous_Liquid.py).
second_viscosity_solid = -(2./3.)*scafflod_like_viscosity
#
#
# Thresholds for the constitutive law 
# cell_pressure(cell fraction)
# adapted from Byrne and Preziosi (2003)
equilibrium_fraction = dolfinx.default_scalar_type(0.65)
decohesion_fraction  = dolfinx.default_scalar_type(0.50)
interaction_fraction = dolfinx.default_scalar_type(0.00)
# 
# Decohesion pressure
decohesion_pressure                = dolfinx.default_scalar_type(-500)
# 
# Permeability of the interstitium [m²] (plays significant role on v identified)
permea_eps_square = True
if permea_eps_square:
	permeability = dolfinx.default_scalar_type((1e-9)/((1-equilibrium_fraction)*(1-equilibrium_fraction)))
else:
	permeability = dolfinx.default_scalar_type(1e-9)
# 
# --------------------------------------- #
# 	   		 Boundary values			  #
# --------------------------------------- #
# 
# Pressures [Pa]
p_right = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0.))

p_left  = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(1000.))
# v_left  = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(1e-6))
# 
# --------------------------------------- #
# 	   			Operators				  #
# --------------------------------------- #
# Coordinates, for axisymmetry
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
# Mixed Space (R,R,R2) -> (pl,pb,u)
P2_v      = basix.ufl.element("P", domain.topology.cell_name(), degree=2, shape=(domain.topology.dim,))
P1_v      = basix.ufl.element("P", domain.topology.cell_name(), degree=1, shape=(domain.topology.dim,))
P1        = basix.ufl.element("P", domain.topology.cell_name(), degree=1)
# 
# Function spaces
P2v_space     = dolfinx.fem.functionspace(domain, P2_v)
P1v_space     = dolfinx.fem.functionspace(domain, P1_v)
P1_space      = dolfinx.fem.functionspace(domain, P1)
# Mixed-space
MS            = dolfinx.fem.functionspace(mesh=domain, element=basix.ufl.mixed_element([P1,P1,P2_v]))
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
Pressure_, Pressure_to_MS   = MS.sub(0).collapse()
Fraction_, Fraction_to_MS   = MS.sub(1).collapse()
Velocity_, Velocity_to_MS   = MS.sub(2).collapse()
# 
# Create the test functions
q_p, q_e, q_v = ufl.TestFunctions(MS)
# 
# --------------------------------------- #
# 	   		   Expressions			      #
# --------------------------------------- #
porosity_expr      = dolfinx.fem.Expression(1-previous_solution.sub(1) , P1_space.element.interpolation_points())
cell_pressure_expr = dolfinx.fem.Expression(constitutive_laws.Byrne_law(previous_solution.sub(1),equilibrium_fraction,interaction_fraction,decohesion_fraction,decohesion_pressure,log=True) , P1_space.element.interpolation_points())
# --------------------------------------- #
# 	   		Initial conditions			  #
# --------------------------------------- #
# 
solution.x.array[Fraction_to_MS]=dolfinx.default_scalar_type(equilibrium_fraction)
solution.x.scatter_forward()
previous_solution.x.array[Fraction_to_MS]=dolfinx.default_scalar_type(equilibrium_fraction)
previous_solution.x.scatter_forward()
# 
# --------------------------------------- #
# 	   		Boundary conditions			  #
# --------------------------------------- #
bcs    = []
fdim   = domain.topology.dim - 1
# 
# bottom, right, top, left = 1, 2, 3, 4
# 
# bottom
facets = facet_tags.find(1)
# (vy = 0, tx = 0, grad(pl).n = 0)
# & no slip
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(0), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(0)))
# 
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
# right
facets = facet_tags.find(2)
# Free-draining outlet: fluid pushed in from the left leaks out here.
# vx is intentionally left free (open edge, no horizontal constraint);
# only vy=0 (no vertical slip through the outlet) and pl=0 (drained,
# reservoir-like fluid pressure) are imposed. The vertical branch is the
# dead end where consolidation/trapping happens, not this boundary.
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
#
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(0), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(p_right, dofs, MS.sub(0)))
right_pressure_dofs = dofs
# 
# top
facets = facet_tags.find(3)
# No-slip condition
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(0), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(0)))
# 
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1))) 
# 
# left vy=0
facets = facet_tags.find(4)
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
# 
# --------------------------------------- #
# 	   			Weak Form				  #
# --------------------------------------- #
# 
fluid_pressure,  cell_fraction,  cell_velocity  = ufl.split(solution)
fluid_pressure_n,  cell_fraction_n,  cell_velocity_n = ufl.split(previous_solution)
#  
# Weak form
# mass epsilon c
F = (
	(1/dt)*(cell_fraction-cell_fraction_n)*q_e*dx 
	+ ufl.nabla_div(cell_fraction*cell_velocity)*q_e*dx
	)
# mass epsilon m
F+= (
	(1/dt)*(cell_fraction_n-cell_fraction)*q_p*dx 
	+ ufl.nabla_div((1-cell_fraction)*cell_velocity)*q_p*dx 
	)
if permea_eps_square == False:
	F+= ((permeability)/interstitial_fluid_viscosity)*ufl.dot(ufl.grad(fluid_pressure), ufl.grad(q_p))*dx
else:
	F+= ((permeability*(1-cell_fraction)**2)/interstitial_fluid_viscosity)*ufl.dot(ufl.grad(fluid_pressure), ufl.grad(q_p))*dx
	
# test velocity
F+=(
	- fluid_pressure*ufl.nabla_div(q_v)*dx
	-  (constitutive_laws.Byrne_law(cell_fraction,equilibrium_fraction,interaction_fraction,decohesion_fraction,decohesion_pressure,log=True))*ufl.nabla_div(q_v)*dx
	+ scafflod_like_viscosity*ufl.inner(ufl.grad(cell_velocity), ufl.grad(q_v))*dx
	+ scafflod_like_viscosity*ufl.inner(ufl.grad(cell_velocity).T, ufl.grad(q_v))*dx
	+ second_viscosity_solid*ufl.nabla_div(cell_velocity)*ufl.nabla_div(q_v)*dx
	+ p_left*ufl.dot(normal, q_v)*ds(4)
	)
# 
# --------------------------------------- #
# 	   				Solver			 	  #
# --------------------------------------- #
# Non linear problem definition
dsolution = ufl.TrialFunction(MS)
J         = ufl.derivative(F, solution, dsolution)
Problem   = NonlinearProblem(F, solution, bcs = bcs, J = J)
solver    = functions.set_non_linear_solver_parameters(domain, Problem, 1e-10, 1e-12, "incremental", 10, log_newton=log)
# 
# --------------------------------------- #
# 	   	  Solving & Post-treatment		  #
# --------------------------------------- #
# 
pressuref_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressuref_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressuref_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressuref_space3       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressuref_space4       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressures_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressures_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressures_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressures_space3       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressures_space4       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
L_velocity_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
L_velocity_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
L_velocity_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
L_velocity_space3       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
L_velocity_space4       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
R_velocity_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
R_velocity_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
R_velocity_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
R_velocity_space3       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
R_velocity_space4       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
#
# Continuous time-history tracking (every step), used to diagnose the
# actual transient timescale -- the n0..n4 spatial snapshots above are
# single-instant profiles, not a time history, and are of no help for
# choosing when the transient actually happens.
#   pressuref_inlet_t : fluid pressure at x=0 (loaded end)
#   pressuref_mid_t   : fluid pressure at x=L1/2 (mid-channel)
#   L_velocity_mid_t  : solid velocity (x-comp) at (x=L1/10, y=H1/2)
#   v_branch_t        : solid velocity (y-comp, "infiltration") at the
#                        branch midpoint (x=L1/2, y=H1+H2/2)
t_history          = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
pressuref_inlet_t  = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
pressuref_mid_t    = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
L_velocity_mid_t   = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
v_branch_t         = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
#
# Convergence-study time histories, evaluated on the P2 velocity (no projection)
velocity_P2        = dolfinx.fem.Function(Velocity_)
conv_vx_inlet_t    = numpy.full((num_steps+1, num_conv), numpy.nan)
conv_vx_outlet_t   = numpy.full((num_steps+1, num_conv), numpy.nan)
conv_vy_branch_t   = numpy.full(num_steps+1, numpy.nan)
completed_steps    = 0
def record_convergence_probes(index):
	velocity_P2.x.array[:] = previous_solution.x.array[Velocity_to_MS]
	velocity_P2.x.scatter_forward()
	vals = evaluate_points(domain, velocity_P2, conv_colliding_cells, conv_points)
	if domain.comm.rank == 0:
		conv_vx_inlet_t[index]  = vals[:num_conv, 0]
		conv_vx_outlet_t[index] = vals[num_conv:2*num_conv, 0]
		conv_vy_branch_t[index] = vals[2*num_conv, 1]
#
#
# Open file for export
xdmf = dolfinx.io.XDMFFile(domain.comm, folder+"Output.xdmf", "w")
xdmf.write_mesh(domain)
# vtk outputs
pressure_vtk       = dolfinx.io.VTKFile(domain.comm, folder+"VTK/pressure.pvd", "w")
pressure_vtk.write_mesh(domain)
fraction_vtk       = dolfinx.io.VTKFile(domain.comm, folder+"VTK/fraction.pvd", "w")
fraction_vtk.write_mesh(domain)
velocity_vtk       = dolfinx.io.VTKFile(domain.comm, folder+"VTK/velocity.pvd", "w")
velocity_vtk.write_mesh(domain)
porosity_vtk       = dolfinx.io.VTKFile(domain.comm, folder+"VTK/porosity.pvd", "w")
porosity_vtk.write_mesh(domain)
cell_pressure_vtk  = dolfinx.io.VTKFile(domain.comm, folder+"VTK/cell_pressure.pvd", "w")
cell_pressure_vtk.write_mesh(domain)
# initialise time
t = 0
__pressure, __fraction, __v  = previous_solution.split()
__pressure.name              = "Fluid Pressure"
__fraction.name              = "Cell Fraction"
__pressure2 = __pressure.collapse()
__fraction2 = __fraction.collapse()
__pressure2.name              = "Fluid Pressure"
__fraction2.name              = "Cell Fraction"
velocity.interpolate(previous_solution.sub(2))
velocity.x.scatter_forward()
velocity.name                = "Cell Velocity"
porosity.interpolate(porosity_expr)
porosity.x.scatter_forward()
porosity.name                = "Porosity"
cell_pressure.interpolate(cell_pressure_expr)
cell_pressure.x.scatter_forward()
cell_pressure.name                = "cell pressure"
# Export in the xdmf file 
xdmf.write_function(__pressure,t)
xdmf.write_function(__fraction,t)
xdmf.write_function(velocity,t)
xdmf.write_function(porosity,t)
xdmf.write_function(cell_pressure,t)
pressure_vtk.write_function(__pressure2,t)
fraction_vtk.write_function(__fraction2,t)
velocity_vtk.write_function(velocity,t)
porosity_vtk.write_function(porosity,t)
cell_pressure_vtk.write_function(cell_pressure,t)
# Initial (t=0) point in the continuous time-history tracking
evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+0), points[2*num_points+0], pressuref_inlet_t, 0)
evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+7), points[2*num_points+7], pressuref_mid_t, 0)
evaluate_point(domain, velocity.sub(0), colliding_cells.links(7), points[7], L_velocity_mid_t, 0)
evaluate_point(domain, velocity.sub(1), branch_colliding_cells.links(0), branch_point[0], v_branch_t, 0)
record_convergence_probes(0)
#
for iteration in range(num_steps):
	t = (iteration+1)*dt
	if mpi4py.MPI.COMM_WORLD.rank == 0:
		print('Step: ',iteration+1,'/',num_steps, 'time:', t, ' s')
	try:
		num_its, converged = solver.solve(solution)
		if mpi4py.MPI.COMM_WORLD.rank == 0:
			print(f'   Newton iterations: {num_its}, converged: {converged}')
			rp_vals = solution.x.array[right_pressure_dofs]
			# rank 0 may own no right-boundary dof when the mesh is partitioned
			if rp_vals.size > 0:
				print(f'   right-boundary pressure DOFs: min={rp_vals.min():.6g}, max={rp_vals.max():.6g} (should be exactly {float(p_right.value)})')
	except Exception as e:
		if mpi4py.MPI.COMM_WORLD.rank == 0:
			print("*************")
			print("Solver failed")
			print("*************")
			import traceback
			traceback.print_exc()
			import numpy
			frac_vals = solution.x.array[Fraction_to_MS]
			print(f"[diagnostic] cell_fraction in `solution` at failure: "
				  f"min={numpy.nanmin(frac_vals):.4f}, max={numpy.nanmax(frac_vals):.4f}, "
				  f"any_nan={numpy.isnan(frac_vals).any()}")
		break
	# Ensure pushing the solution to all processes
	solution.x.scatter_forward()
	# Update previous solution
	previous_solution.x.array[:] = solution.x.array[:]
	previous_solution.x.scatter_forward()
	# 
	__pressure, __fraction, __v  = previous_solution.split()
	__pressure.name              = "Fluid Pressure"
	__fraction.name              = "Cell Fraction"
	__pressure2 = __pressure.collapse()
	__fraction2 = __fraction.collapse()
	__pressure2.name              = "Fluid Pressure"
	__fraction2.name              = "Cell Fraction"
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
	# Continuous time-history tracking (every step)
	t_history[iteration+1] = t
	evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+0), points[2*num_points+0], pressuref_inlet_t, iteration+1)
	evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+7), points[2*num_points+7], pressuref_mid_t, iteration+1)
	evaluate_point(domain, velocity.sub(0), colliding_cells.links(7), points[7], L_velocity_mid_t, iteration+1)
	evaluate_point(domain, velocity.sub(1), branch_colliding_cells.links(0), branch_point[0], v_branch_t, iteration+1)
	record_convergence_probes(iteration+1)
	completed_steps = iteration+1
	if (not args.no_fields) and iteration%1 == 0:
		# Export in the xdmf file 
		xdmf.write_function(__pressure,t)
		xdmf.write_function(__fraction,t)
		xdmf.write_function(velocity,t)
		xdmf.write_function(porosity,t)
		xdmf.write_function(cell_pressure,t)
		pressure_vtk.write_function(__pressure2,t)
		fraction_vtk.write_function(__fraction2,t)
		velocity_vtk.write_function(velocity,t)
		porosity_vtk.write_function(porosity,t)
		cell_pressure_vtk.write_function(cell_pressure,t)
		# in space
	if iteration == n0:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressuref_space0, ii)
			evaluate_point(domain, cell_pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressures_space0, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(num_points+ii), points[num_points+ii], R_velocity_space0, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(ii), points[ii], L_velocity_space0, ii)
	elif iteration == n1:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressuref_space1, ii)
			evaluate_point(domain, cell_pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressures_space1, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(num_points+ii), points[num_points+ii], R_velocity_space1, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(ii), points[ii], L_velocity_space1, ii)
	elif iteration == n2:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressuref_space2, ii)
			evaluate_point(domain, cell_pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressures_space2, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(num_points+ii), points[num_points+ii], R_velocity_space2, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(ii), points[ii], L_velocity_space2, ii)
	elif iteration == n3:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressuref_space3, ii)
			evaluate_point(domain, cell_pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressures_space3, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(num_points+ii), points[num_points+ii], R_velocity_space3, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(ii), points[ii], L_velocity_space3, ii)
	elif iteration == n4:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressuref_space4, ii)
			evaluate_point(domain, cell_pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressures_space4, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(num_points+ii), points[num_points+ii], R_velocity_space4, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(ii), points[ii], L_velocity_space4, ii)
xdmf.close()
pressure_vtk.close()
fraction_vtk.close()
velocity_vtk.close()
porosity_vtk.close()
cell_pressure_vtk.close()
# 
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if mpi4py.MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
# 
# EoF
n_cells = domain.topology.index_map(domain.topology.dim).size_global
n_dofs  = MS.dofmap.index_map.size_global*MS.dofmap.index_map_bs
if mpi4py.MPI.COMM_WORLD.rank == 0:
	# Raw data of the spatial/temporal convergence study
	# (post-processed by convergence_postprocess.py)
	numpy.savez(os.path.join(folder, "convergence_data.npz"),
		t=t_history[:completed_steps+1], y=y_conv,
		vx_inlet=conv_vx_inlet_t[:completed_steps+1], vx_outlet=conv_vx_outlet_t[:completed_steps+1],
		vy_branch=conv_vy_branch_t[:completed_steps+1],
		dt=dt, t_fin=t_fin, completed=(completed_steps == num_steps),
		size_max=size_max, size_min=size_min,
		mean_h=mesh_stats['mean_h'], std_h=mesh_stats['std_dev_h'],
		n_cells=n_cells, n_dofs=n_dofs)
	functions.export_to_csv(
		[x_check,pressuref_space0,pressures_space0,pressuref_space1,pressures_space1,pressuref_space2,pressures_space2,pressuref_space3,pressures_space3,pressuref_space4,pressures_space4],
		os.path.join(csv_dir, "pressure_space.csv"),
		["x","pressuref_space0","pressures_space0","pressuref_space1","pressures_space1","pressuref_space2","pressures_space2","pressuref_space3","pressures_space3","pressuref_space4","pressures_space4"]
		)
	functions.export_to_csv(
		[y_check,L_velocity_space0,R_velocity_space0,L_velocity_space1,R_velocity_space1,L_velocity_space2,R_velocity_space2,L_velocity_space3,R_velocity_space3,L_velocity_space4,R_velocity_space4],
		os.path.join(csv_dir, "velocity_space.csv"),
		["y","L_velocity_space0","R_velocity_space0","L_velocity_space1","R_velocity_space1","L_velocity_space2","R_velocity_space2","L_velocity_space3","R_velocity_space3","L_velocity_space4","R_velocity_space4"]
		)
	# Diagnostic-only time-history export (not read by any existing tex
	# figure): used to inspect the real transient shape and pick sensible
	# n0..n4 snapshot times, since the 5 spatial snapshots above are single
	# instants and give no information on the time evolution.
	functions.export_to_csv(
		[t_history,pressuref_inlet_t,pressuref_mid_t,L_velocity_mid_t,v_branch_t],
		os.path.join(csv_dir, "time_series.csv"),
		["t","pressuref_inlet","pressuref_mid","L_velocity_mid","v_branch"]
		)
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

	input_file = os.path.join(csv_dir, "pressure_space.csv")
	output_file = os.path.join(csv_dir, "pressure_space_t.csv")
	transpose_csv(input_file,output_file)

	input_file = os.path.join(csv_dir, "velocity_space.csv")
	output_file = os.path.join(csv_dir, "velocity_space_t.csv")
	transpose_csv(input_file,output_file)