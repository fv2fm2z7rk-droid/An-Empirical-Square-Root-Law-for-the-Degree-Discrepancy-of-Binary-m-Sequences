#!/usr/bin/env python3
"""
full_rho_profiles_k28_down.py

Same as full_rho_profiles_k28.py but the grid of relative lengths
is traversed from ρ ≈ 1 down to ρ ≈ 0.
"""

import json
import math
import time
from pathlib import Path
import numpy as np

# ------------------------------------------------------------------
PRIMITIVE_POLY = 0b10000000000000000000000001001   # x^28 + x^3 + 1
K = 28
JSON_PATH = Path("full_rho_profiles_k28_down.json")
N_POINTS = 28
N_SAMPLES = 1200
# ------------------------------------------------------------------

def advance_state(state, k, prim):
    state <<= 1
    if state & (1 << k):
        state ^= prim
    return state & ((1 << k) - 1)

def generate_degree_sequence(k, prim):
    M = (1 << k) - 1
    deg = [0] * M
    s = 1
    for t in range(M):
        deg[t] = s.bit_length() - 1
        s = advance_state(s, k, prim)
    return deg

def sampled_delta(k, r, degrees, n_samples=1200, seed=0):
    M = (1 << k) - 1
    if r >= M:
        return 0.0
    rng = np.random.default_rng(seed + k * 10007 + r)
    n = min(n_samples, M)
    starts = rng.choice(M, size=n, replace=False)
    ideal = [(r * (1 << j)) / M for j in range(k)]
    tvs = []
    for s in starts:
        counts = [0] * k
        for i in range(r):
            counts[degrees[(int(s) + i) % M]] += 1
        tv = sum(abs(counts[j] - ideal[j]) for j in range(k)) / 2.0
        tvs.append(tv)
    return max(tvs) - min(tvs)

def save_partial(data):
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [saved partial profile → {JSON_PATH}]", flush=True)

def main():
    M = (1 << K) - 1
    N = 1 << K
    print(f"=== k={K}  (M={M})  — downward sweep ===")
    print("Generating degree sequence ...", flush=True)
    t0 = time.perf_counter()
    degrees = generate_degree_sequence(K, PRIMITIVE_POLY)
    print(f"Done in {time.perf_counter()-t0:.1f}s\n", flush=True)

    # geometric lower part + linear upper part, then reverse
    rels_lo = np.geomspace(1.0 / M, 0.3, max(2, N_POINTS // 2))
    rels_hi = np.linspace(0.3, 1.0, N_POINTS // 2 + 1)[1:]
    rels = np.unique(np.concatenate([rels_lo, rels_hi]))
    rels = rels[::-1]          # <-- start from ρ≈1 and go downwards

    rs, deltas, rDeltas, rhos = [], [], [], []

    data = {
        "k": K,
        "M": M,
        "N": N,
        "r": rs,
        "rho": rhos,
        "Delta": deltas,
        "rDelta": rDeltas,
        "max_Delta": 0.0,
        "max_rDelta": 0.0,
        "method": "sampled",
        "direction": "downward",
    }

    for rf in rels:
        r = max(1, min(M, int(round(rf * M))))
        t1 = time.perf_counter()
        ns = N_SAMPLES if r < M // 5 else max(400, N_SAMPLES // 2)
        dlt = sampled_delta(K, r, degrees, n_samples=ns)
        elapsed = time.perf_counter() - t1

        rs.append(r)
        rhos.append(r / M)
        deltas.append(float(dlt))
        rDeltas.append(float(r * dlt))
        data["max_Delta"] = max(deltas)
        data["max_rDelta"] = max(rDeltas)

        print(f"  r={r:9d}  ρ={r/M:.4f}  Δ={dlt:.3f}  [{elapsed:.1f}s]", flush=True)
        save_partial(data)

    print(f"\nFinished k={K} (downward)")
    print(f"  max Δ = {data['max_Delta']:.3f}")
    print(f"  ½√N   = {0.5 * math.sqrt(N):.3f}")
    print(f"  d_k   = {data['max_Delta'] / math.sqrt(N):.3f}")
    print(f"Results in {JSON_PATH}")

if __name__ == "__main__":
    main()