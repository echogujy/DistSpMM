#!/usr/bin/env python3
"""Fit the adaptive-communication model parameters used by DistSPMM.

The kernel's adaptive communication selector (`src/comm_man.h`, comm_type=2)
uses a piecewise LogGP-based model:

    threshold(k) = 1 / (a * k^u + b / k)         for k < c
                   1 / (a * c^u + b / c)          for k >= c

where `k` is the embedding dimension (dense matrix columns) and `threshold` is
the critical non-zero-column density above which dense block communication is
cheaper than sparse communication. This script fits (a, b, c, u) separately for
the intra-node (NVLink) and inter-node (InfiniBand) tiers from measured data.

The fitted values are printed and can be copied into `src/comm_man.h`.

Run inside the `datapass` conda environment:
    conda activate datapass
    python3 tools/fit_comm_model.py
"""

import numpy as np
from scipy.optimize import curve_fit

# --- measured threshold data ----------------------------------------------------
# NOTE: The values below are the *critical densities* measured by the authors on
# their own test platforms. They are NOT universal: the thresholds depend on the
# NVLink / InfiniBand bandwidth of your hardware. If you run on a different
# platform, re-measure them yourself and update MEASURED before fitting.
#
# How to re-measure on your platform:
#   1. Build and run the communication microbenchmark on your cluster:
#        cd scripts && sbatch comm_test_slurm.sh
#      (the script loops over N = 400k..2M and dims = 1..2048 with stride 2,
#       once with com_type=0 for the intra-node tier and once with com_type=1
#       for the inter-node tier.)
#   2. Each `comm_test` run prints one line per dim:
#        <dim>,<threshold>
#      where `threshold` is the critical non-zero-column density above which
#      dense block communication beats sparse communication for that dim.
#   3. Collect these lines per N and per tier into the `MEASURED` table below
#      (one tuple per N: (N, intra_thresholds[], inter_thresholds[])).
#
# DIMS must list the embedding dimensions in the same order as the threshold
# arrays in MEASURED.
DIMS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Each entry: (N, intra_node_thresholds[], inter_node_thresholds[]).
# Group 0 (indices 0,2,4,6): N = 400k, 1000k, 1400k, 1800k
# Group 1 (indices 1,3,5,7): N = 800k, 1200k, 1600k, 2000k
MEASURED = [
    # N=400000
    (400000,
     [0, 0, 0, 0.029703, 0.158416, 0.247525, 0.49505, 0.831683, 0.831683, 0.831683, 0.831683],
     [0.049505, 0.0594059, 0.0990099, 0.158416, 0.326733, 0.623762, 0.960396, 0.990099, 0.990099, 0.70297, 0.782178]),
    # N=800000
    (800000,
     [0, 0, 0.039604, 0.0594059, 0.118812, 0.247525, 0.49505, 0.831683, 0.831683, 0.851485, 0.851485],
     [0.029703, 0.039604, 0.0792079, 0.158416, 0.316832, 0.613861, 0.940594, 0.990099, 0.990099, 0.722772, 0.782178]),
    # N=1000000
    (1000000,
     [0, 0, 0.029703, 0.0594059, 0.128713, 0.267327, 0.49505, 0.831683, 0.831683, 0.851485, 0.861386],
     [0.029703, 0.049505, 0.0792079, 0.158416, 0.316832, 0.613861, 0.930693, 0.990099, 0.990099, 0.712871, 0.782178]),
    # N=1200000
    (1200000,
     [0, 0, 0.019802, 0.049505, 0.138614, 0.247525, 0.49505, 0.831683, 0.851485, 0.861386, 0.861386],
     [0.019802, 0.039604, 0.0792079, 0.158416, 0.316832, 0.60396, 0.940594, 0.990099, 0.990099, 0.70297, 0.792079]),
    # N=1400000
    (1400000,
     [0, 0, 0.019802, 0.0693069, 0.138614, 0.257426, 0.524752, 0.821782, 0.851485, 0.851485, 0.851485],
     [0.019802, 0.039604, 0.0792079, 0.158416, 0.306931, 0.60396, 0.950495, 0.990099, 0.990099, 0.732673, 0.792079]),
    # N=1600000
    (1600000,
     [0, 0.019802, 0.019802, 0.0594059, 0.128713, 0.267327, 0.504951, 0.831683, 0.851485, 0.851485, 0.861386],
     [0.019802, 0.039604, 0.0792079, 0.158416, 0.306931, 0.60396, 0.950495, 0.990099, 0.990099, 0.722772, 0.80198]),
    # N=1800000
    (1800000,
     [0, 0.00990099, 0.029703, 0.0693069, 0.128713, 0.257426, 0.514852, 0.831683, 0.851485, 0.851485, 0.861386],
     [0.019802, 0.039604, 0.0792079, 0.158416, 0.306931, 0.60396, 0.950495, 0.990099, 0.990099, 0.732673, 0.80198]),
    # N=2000000
    (2000000,
     [0, 0.00990099, 0.029703, 0.0594059, 0.128713, 0.267327, 0.514852, 0.831683, 0.851485, 0.851485, 0.861386],
     [0.019802, 0.039604, 0.0792079, 0.158416, 0.306931, 0.613861, 0.960396, 0.990099, 0.990099, 0.732673, 0.792079]),
]


def piecewise_function(x, a, b, c, u):
    """Piecewise critical-density model used in the kernel.

    `ponytail:` invalid powers (0^u for u<0, or a transiently non-positive `c`
    during fitting) are harmless for curve_fit; we silence them here.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.piecewise(
            x,
            [x < c, x >= c],
            [lambda x: 1 / (a * (x ** u) + b / x),
             lambda x: 1 / (a * (c ** u) + b / c)],
        )


def fit_model(x_data, y_data):
    """Fit (a, b, c, u) of the piecewise model to (x_data, y_data)."""
    params, _ = curve_fit(piecewise_function, x_data, y_data,
                          p0=[1, 1, 100, 1], maxfev=10000)
    return params


def fit_all():
    """Fit intra-node and inter-node parameters from the measured data."""
    x_data = np.asarray(DIMS, dtype=np.float64)

    # Two groups interleaved by N; fit on the group-0 average (as in the paper).
    group_0_indices = [0, 2, 4, 6]
    intra_group_0 = [MEASURED[i][1] for i in group_0_indices]
    inter_group_0 = [MEASURED[i][2] for i in group_0_indices]
    y_intra = np.mean(intra_group_0, axis=0)
    y_inter = np.mean(inter_group_0, axis=0)

    a_intra, b_intra, c_intra, u_intra = fit_model(x_data, y_intra)
    a_inter, b_inter, c_inter, u_inter = fit_model(x_data, y_inter)

    return {
        "intra": (a_intra, b_intra, c_intra, u_intra),
        "inter": (a_inter, b_inter, c_inter, u_inter),
    }


def main():
    params = fit_all()
    a_intra, b_intra, c_intra, u_intra = params["intra"]
    a_inter, b_inter, c_inter, u_inter = params["inter"]

    print("Intra-node (NVLink) Parameters:")
    print(f"  a = {a_intra:.5f}, b = {b_intra:.5f}, c = {c_intra:.5f}, u = {u_intra:.2f}")
    print("Inter-node (InfiniBand) Parameters:")
    print(f"  a = {a_inter:.5f}, b = {b_inter:.5f}, c = {c_inter:.5f}, u = {u_inter:.2f}")
    print()
    print("Copy these into src/comm_man.h (comm_type == 2 branch):")
    print(f"  intra: a={a_intra:.5f}, b={b_intra:.5f}, c={c_intra:.5f}, u={u_intra:.2f}")
    print(f"  inter: a={a_inter:.5f}, b={b_inter:.5f}, c={c_inter:.5f}, u={u_inter:.2f}")


if __name__ == "__main__":
    main()
