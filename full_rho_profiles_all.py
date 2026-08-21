#!/usr/bin/env python3
"""
full_rho_profiles_all.py

Recompute full-horizon discrepancy *profiles* for k = 2..28.
- Exact sliding-window evaluation for k <= 14
- Monte-Carlo sampling for k >= 15
- All polynomials are degree-checked at startup
- Results are profile maxima, not exhaustive maxima over every r
"""

import argparse
import json
import math
import time
from pathlib import Path
import numpy as np

# ------------------------------------------------------------------
# Degree-correct primitive polynomials (bit_length()-1 == k)
# ------------------------------------------------------------------
PRIMITIVE_POLYNOMIALS = {
    2:  0b111,                                  # x^2 + x + 1
    3:  0b1011,                                 # x^3 + x + 1
    4:  0b10011,                                # x^4 + x + 1
    5:  0b100101,                               # x^5 + x^2 + 1
    6:  0b1000011,                              # x^6 + x + 1
    7:  0b10000011,                             # x^7 + x + 1
    8:  0b100011101,                            # x^8 + x^4 + x^3 + x^2 + 1
    9:  0b1000010001,                           # x^9 + x^4 + 1
    10: 0b10000001001,                          # x^10 + x^3 + 1
    11: 0b100000000101,                         # x^11 + x^2 + 1
    12: 0b1000001010011,                        # x^12 + x^6 + x^4 + x + 1
    13: 0b10000000011011,                       # x^13 + x^4 + x^3 + x + 1
    14: 0b100000000101011,                      # x^14 + x^5 + x^3 + x + 1
    15: 0b1000000000000011,                     # x^15 + x + 1
    16: 0b10000000000101101,                    # x^16 + x^5 + x^3 + x^2 + 1
    17: 0b100000000000001001,                   # x^17 + x^3 + 1
    18: 0b1000000000010000001,                  # x^18 + x^7 + 1
    19: 0b10000000000000100111,                 # x^19 + x^5 + x^2 + x + 1
    20: 0b100000000000000001001,                # x^20 + x^3 + 1
    21: 0b1000000000000000000101,               # x^21 + x^2 + 1
    22: 0b10000000000000000000011,              # x^22 + x + 1
    23: 0b100000000000000000100001,             # x^23 + x^5 + 1
    24: 0b1000000000000000010000111,            # x^24 + x^7 + x^2 + x + 1
    25: 0b10000000000000000000001001,           # x^25 + x^3 + 1
    26: 0b100000000000000000001000111,          # x^26 + x^6 + x^2 + x + 1
    27: 0b1000000000000000000000100111,         # x^27 + x^5 + x^2 + x + 1
    28: 0b10000000000000000000000001001,        # x^28 + x^3 + 1
}

def validate_polynomials():
    print("Validating polynomial degrees ...")
    for k, p in sorted(PRIMITIVE_POLYNOMIALS.items()):
        deg = p.bit_length() - 1
        if deg != k:
            raise ValueError(f"Polynomial for k={k} has degree {deg}")
    print("All polynomials have the correct degree.\n")

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

def sliding_delta(k, r, degrees):
    """Exact range of the normalised l1-deviation over all contiguous windows."""
    M = (1 << k) - 1
    if r >= M:
        return 0.0
    ideal = [(r * (1 << j)) / M for j in range(k)]
    counts = [0] * k
    for i in range(r):
        counts[degrees[i]] += 1
    tv = sum(abs(counts[j] - ideal[j]) for j in range(k))
    min_tv = max_tv = tv
    for t in range(1, M):
        out = degrees[t - 1]
        inn = degrees[(t - 1 + r) % M]
        if out != inn:
            tv -= abs(counts[out] - ideal[out])
            counts[out] -= 1
            tv += abs(counts[out] - ideal[out])
            tv -= abs(counts[inn] - ideal[inn])
            counts[inn] += 1
            tv += abs(counts[inn] - ideal[inn])
        if tv < min_tv:
            min_tv = tv
        if tv > max_tv:
            max_tv = tv
    return (max_tv - min_tv) / 2.0

def sampled_delta(k, r, degrees, n_samples=1200, seed=0):
    """Monte-Carlo estimate of the range (conservative)."""
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

def profile_full(k, n_points=28, n_samples=1200):
    M = (1 << k) - 1
    prim = PRIMITIVE_POLYNOMIALS[k]
    print(f"  generating degree sequence (M={M}) ...", flush=True)
    t0 = time.perf_counter()
    degrees = generate_degree_sequence(k, prim)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)

    # geometric lower part + linear upper part
    rels_lo = np.geomspace(1.0 / M, 0.3, max(2, n_points // 2))
    rels_hi = np.linspace(0.3, 1.0, n_points // 2 + 1)[1:]
    rels = np.unique(np.concatenate([rels_lo, rels_hi]))

    rs, deltas, rDeltas = [], [], []
    use_exact = (k <= 14)

    for rf in rels:
        r = max(1, min(M, int(round(rf * M))))
        t1 = time.perf_counter()
        if use_exact:
            dlt = sliding_delta(k, r, degrees)
            mode = "exact"
        else:
            ns = n_samples if r < M // 5 else max(400, n_samples // 2)
            dlt = sampled_delta(k, r, degrees, n_samples=ns)
            mode = "sampled"
        rs.append(r)
        deltas.append(float(dlt))
        rDeltas.append(float(r * dlt))
        print(f"    r={r:9d}  ρ={r/M:.4f}  Δ={dlt:.3f}  [{mode}, {time.perf_counter()-t1:.1f}s]",
              flush=True)

    return {
        "k": k,
        "M": M,
        "N": 1 << k,
        "r": rs,
        "rho": [r / M for r in rs],
        "Delta": deltas,
        "rDelta": rDeltas,
        "max_Delta": max(deltas),
        "max_rDelta": max(rDeltas),
        "method": "exact" if use_exact else "sampled",
    }

def main():
    parser = argparse.ArgumentParser(
        description="Recompute full-rho profiles for k=2..28 (profile maxima)"
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=28)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--points", type=int, default=28)
    parser.add_argument("--json", type=str, default="full_rho_profiles.json")
    args = parser.parse_args()

    validate_polynomials()

    path = Path(args.json)
    results = {}
    if path.exists():
        with open(path) as f:
            results = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded existing data for k = {sorted(results.keys())}")

    t_global = time.perf_counter()

    for k in range(args.k_min, args.k_max + 1):
        if k not in PRIMITIVE_POLYNOMIALS:
            print(f"Skipping k={k} (no polynomial)")
            continue
        print(f"\n=== k={k} ===")
        results[k] = profile_full(k, n_points=args.points, n_samples=args.samples)

        # save after every k
        with open(path, "w") as f:
            json.dump({str(kk): vv for kk, vv in sorted(results.items())}, f, indent=2)

        md = results[k]["max_Delta"]
        N = results[k]["N"]
        print(f"  → max Δ = {md:.3f}   ½√N = {0.5*math.sqrt(N):.3f}   "
              f"d_k = {md / math.sqrt(N):.3f}")

    print(f"\nAll done in {time.perf_counter()-t_global:.1f}s")
    print(f"Results written to {path}")
    print(f"k values present: {sorted(results.keys())}")

if __name__ == "__main__":
    main()