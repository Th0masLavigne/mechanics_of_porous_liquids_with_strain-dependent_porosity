# Post-processing of the spatial and temporal convergence study of the
# bifurcated-channel simulation (run_convergence_study.sh).
#
# Metrics (v = x-component of the liquid-scaffold velocity, P2 field):
#   * snapshot error at a time t* on the sections x = 0.1L and x = 0.9L:
#         e(t*) = || v_h(.,t*) - v_ref(.,t*) ||_{L2(0,H)} / || v_ref(.,t*) ||_{L2(0,H)}
#   * space-time error over the whole simulation (both sections together):
#         E = sqrt( sum_n ||v_h(.,t_n) - v_ref(.,t_n)||^2 / sum_n ||v_ref(.,t_n)||^2 )
#   * branch infiltration: relative L2-in-time error of v_y(t) at the branch
#     mid-point and relative error on its peak value
#   * observed order of convergence from three successive levels:
#         p = ln( E_{k-1,k} / E_{k,k+1} ) / ln(r)
#     with E_{i,j} the space-time difference between levels i and j and r the
#     refinement ratio (mean element size or time step).
# The reference solution is the finest level of each series.
#
# Outputs (folder convergence_results/)
import argparse
import os
import numpy

SNAPSHOTS = (0.01, 1.15, 5.0)

def trapz(f, y):
	return numpy.trapezoid(f, y) if hasattr(numpy, "trapezoid") else numpy.trapz(f, y)

def ynorm(f, y):
	return numpy.sqrt(trapz(f**2, y))

def load(folder):
	path = os.path.join(folder, "convergence_data.npz")
	run  = dict(numpy.load(path))
	if not bool(run["completed"]):
		print(f"WARNING: {path} did not reach t_fin (solver failure?)")
	run["folder"] = folder
	return run

def interp_time(run, key, t):
	"""Linear interpolation in time of a stored history (None if t is not covered by a time step)."""
	tt = run["t"]
	if t < tt[1] - 1e-12 or t > tt[-1] + 1e-12:
		return None
	arr = run[key]
	if arr.ndim == 1:
		return numpy.interp(t, tt, arr)
	return numpy.array([numpy.interp(t, tt, arr[:, j]) for j in range(arr.shape[1])])

def snapshot_error(run, ref, key, t):
	v, r = interp_time(run, key, t), interp_time(ref, key, t)
	if v is None or r is None:
		return numpy.nan
	return ynorm(v - r, run["y"]) / ynorm(r, run["y"])

def spacetime_diff(run, ref):
	"""Relative space-time difference of the profiles of `run` w.r.t. `ref`, on the time grid of `run`."""
	y, num, den = run["y"], 0., 0.
	for n, t in enumerate(run["t"][1:], start=1):
		if t > ref["t"][-1] + 1e-12:
			break
		for key in ("vx_inlet", "vx_outlet"):
			r    = interp_time(ref, key, t)
			num += ynorm(run[key][n] - r, y)**2
			den += ynorm(r, y)**2
	return numpy.sqrt(num/den)

def branch_errors(run, ref):
	tt  = run["t"][1:]
	tt  = tt[tt <= ref["t"][-1] + 1e-12]
	v   = run["vy_branch"][1:len(tt)+1]
	r   = numpy.interp(tt, ref["t"], ref["vy_branch"])
	err = numpy.sqrt(numpy.sum((v - r)**2) / numpy.sum(r**2))
	i_v = numpy.argmax(numpy.abs(run["vy_branch"]))
	i_r = numpy.argmax(numpy.abs(ref["vy_branch"]))
	peak_v, peak_r = run["vy_branch"][i_v], ref["vy_branch"][i_r]
	return err, abs(peak_v - peak_r)/abs(peak_r), peak_v, run["t"][i_v]

def fmt(x):
	return "" if (x is None or (isinstance(x, float) and numpy.isnan(x))) else f"{x:.6e}"

def write_csv(path, header, rows):
	with open(path, "w") as f:
		f.write(",".join(header) + "\n")
		for row in rows:
			f.write(",".join(fmt(float(x)) if not isinstance(x, str) else x for x in row) + "\n")

def analyse(series, runs, param, ratio_key, out_dir, lines):
	ref  = runs[-1]
	n    = len(runs)
	# successive differences for the observed order
	succ = [spacetime_diff(runs[k], runs[k+1]) for k in range(n-1)]
	header = [param, "std_h_mm", "n_cells", "n_dofs", "n_steps", "E_st"]
	header += [f"Ein_{int(round(t*100)):03d}"  for t in SNAPSHOTS]
	header += [f"Eout_{int(round(t*100)):03d}" for t in SNAPSHOTS]
	header += ["E_branch", "E_peak", "peak_vy", "t_peak", "p_obs"]
	rows = []
	lines.append(f"\n=== {series} study (reference: finest level, {ratio_key} = {float(ref[ratio_key]):.4g}) ===")
	for k, run in enumerate(runs):
		x     = float(run["mean_h"])*1e3 if ratio_key == "mean_h" else float(run["dt"])
		E_st  = spacetime_diff(run, ref) if k < n-1 else 0.
		e_in  = [snapshot_error(run, ref, "vx_inlet",  t) if k < n-1 else 0. for t in SNAPSHOTS]
		e_out = [snapshot_error(run, ref, "vx_outlet", t) if k < n-1 else 0. for t in SNAPSHOTS]
		e_b, e_pk, pk, tpk = branch_errors(run, ref) if k < n-1 else (0., 0., *branch_errors(run, ref)[2:])
		p_obs = numpy.nan
		if 1 <= k <= n-2:
			r     = float(runs[k-1][ratio_key]) / float(runs[k][ratio_key])
			p_obs = numpy.log(succ[k-1]/succ[k]) / numpy.log(r)
		rows.append([x, float(run["std_h"])*1e3, int(run["n_cells"]), int(run["n_dofs"]), len(run["t"])-1, E_st,
					 *e_in, *e_out, e_b, e_pk, pk, tpk, p_obs])
		lines.append(f"{param}={x:.4g}: E_st={E_st:.3e}, e_in(t)={['%.2e' % e for e in e_in]}, "
					 f"e_out(t)={['%.2e' % e for e in e_out]}, E_branch={e_b:.2e}, E_peak={e_pk:.2e}, p_obs={p_obs:.2f}")
	write_csv(os.path.join(out_dir, f"convergence_{series}.csv"), header, rows)
	# velocity profiles at the snapshot times (for the overlay figure)
	for t in SNAPSHOTS:
		cols, head = [runs[0]["y"]], ["y"]
		for key, lab in (("vx_inlet", "in"), ("vx_outlet", "out")):
			for k, run in enumerate(runs):
				v = interp_time(run, key, t)
				cols.append(v if v is not None else numpy.full_like(runs[0]["y"], numpy.nan))
				head.append(f"{lab}{k}")
		write_csv(os.path.join(out_dir, f"profiles_{series}_t{int(round(t*100)):03d}.csv"), head, numpy.array(cols).T.tolist())
	return rows

def quicklook(mesh_runs, dt_runs, out_dir):
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ImportError:
		return
	fig, ax = plt.subplots(2, 3, figsize=(13, 7))
	for row, (runs, lab, fmt_lab) in enumerate(((mesh_runs, "mean_h", r"$\bar h$={:.2f} mm"), (dt_runs, "dt", r"$\Delta t$={:g} s"))):
		for run in runs:
			val   = float(run[lab])*(1e3 if lab == "mean_h" else 1.)
			label = fmt_lab.format(val)
			for col, key in enumerate(("vx_inlet", "vx_outlet")):
				v = interp_time(run, key, 1.15)
				if v is not None:
					ax[row, col].plot(v, run["y"], label=label)
				ax[row, col].set_title(f"$v_x$ at x={'0.1' if col == 0 else '0.9'}L, t=1.15 s")
				ax[row, col].set_xlabel("$v_x$ (m/s)"); ax[row, col].set_ylabel("y (m)")
			ax[row, 2].plot(run["t"], run["vy_branch"], label=label)
			ax[row, 2].set_title("$v_y$ at the branch mid-point"); ax[row, 2].set_xlabel("t (s)")
		ax[row, 0].legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, "convergence_quicklook.png"), dpi=150)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--mesh_tags", nargs="+", default=["4e-3", "2e-3", "1e-3", "5e-4"], help="size_max values, coarse -> fine")
	parser.add_argument("--dt_tags",   nargs="+", default=["4e-2", "2e-2", "1e-2", "5e-3", "2.5e-3"], help="dt values, coarse -> fine")
	parser.add_argument("--out", default="convergence_results")
	args = parser.parse_args()
	os.makedirs(args.out, exist_ok=True)
	mesh_runs = [load(f"outputs_croix_conv_h{h}")   for h in args.mesh_tags]
	dt_runs   = [load(f"outputs_croix_conv_dt{d}") for d in args.dt_tags]
	lines = []
	analyse("mesh", mesh_runs, "h_mm", "mean_h", args.out, lines)
	analyse("dt",   dt_runs,   "dt",   "dt",     args.out, lines)
	text = "\n".join(lines)
	print(text)
	with open(os.path.join(args.out, "convergence_summary.txt"), "w") as f:
		f.write(text + "\n")
	quicklook(mesh_runs, dt_runs, args.out)
