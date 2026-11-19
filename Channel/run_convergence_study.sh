#!/usr/bin/env bash
# Spatial and temporal convergence study of the bifurcated-channel simulation
# (Supplementary Material).
#
#  - spatial study : dt fixed to the reference value (0.01 s),
#                    Gmsh size_max = 4, 2, 1 (reference), 0.5 mm
#  - temporal study: mesh fixed to the reference one (size_max = 1 mm),
#                    dt = 0.04, 0.02, 0.01 (reference), 0.005, 0.0025 s
#
# Usage (inside the dolfinx v0.9.0 container, from the Channel/ folder):
#     NP=4 bash run_convergence_study.sh
#     python3 convergence_postprocess.py
set -euo pipefail
NP=${NP:-4}

MESH_SIZES=(4e-3 2e-3 1e-3 5e-4)
TIME_STEPS=(4e-2 2e-2 1e-2 5e-3 2.5e-3)

for h in "${MESH_SIZES[@]}"; do
    echo "=== spatial study: size_max = ${h} m, dt = 1e-2 s ==="
    mpirun -n "${NP}" python3 Croix_porous_liquid.py --size_max "${h}" --dt 1e-2 \
        --tag "_conv_h${h}" --no_fields
done

for dt in "${TIME_STEPS[@]}"; do
    echo "=== temporal study: dt = ${dt} s, size_max = 1e-3 m ==="
    mpirun -n "${NP}" python3 Croix_porous_liquid.py --size_max 1e-3 --dt "${dt}" \
        --tag "_conv_dt${dt}" --no_fields
done
