#!/usr/bin/env python3
"""
Three-body (F3) and four-body (F4) order parameters for water
-------------------------------------------------------------
CB4103 project: implementation of the water order parameters of
Rodger, Forester & Smith, Fluid Phase Equilibria 116, 326 (1996).

  F3 = < [ |cos(theta)| cos(theta) + cos^2(109.47 deg) ]^2 >
       theta = O_j - O_i - O_k angle, j,k = two oxygens in the first
       solvation shell (r_OO < rc) of oxygen i.   (Clancy / Baez-Clancy)
       -> 0 for a perfect tetrahedral network, grows with disorder.

  F4 = < cos(3 phi) >
       phi = H - O ... O - H torsion of two neighbouring waters, using the
       outer-most hydrogen of each molecule of the dimer.
       -> ~ 1 for the (eclipsed, planar-ring) hydrate lattice, ~ 0 in liquid.

For every water molecule the angles/torsions it takes part in are averaged
first (per-molecule value), then averaged over molecules (per-frame value,
or per x-bin to get a profile across the interface).

Usage
-----
    python order_parameters_F3_F4.py trj_HTR.gro                # full run
    python order_parameters_F3_F4.py trj_HTR.gro --stride 10    # quick look
    python order_parameters_F3_F4.py --selftest                 # unit tests

Outputs (in --out directory)
----------------------------
    F3_F4_vs_time.csv      frame-averaged F3, F4 and number of waters vs time
    F3_F4_profile_x.csv    time-averaged profile along the long box axis
    F3_F4_xt.npz           per-frame x-profiles (for the heat maps)
    F3_F4_summary.png      time series, profile, heat maps, distributions
"""
import argparse
import os
import re
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

COS_TETRA = np.cos(np.radians(109.47))      # cos(109.47 deg) ~ -1/3


# ----------------------------------------------------------------------------
# 1. GRO reader (streaming: only one frame in memory at a time)
# ----------------------------------------------------------------------------
def iter_gro(path, stride=1, first_names=True):
    """Yield (time_ps, box(3,), coords(N,3) in nm, names or None) per frame."""
    tre = re.compile(r"t=\s*([-+0-9.eE]+)")
    with open(path, "rb") as f:
        iframe = 0
        names = None
        while True:
            title = f.readline()
            if not title:
                break
            n = int(f.readline())
            raw = [f.readline() for _ in range(n)]
            box = np.array(f.readline().split(), dtype=float)[:3]  # orthorhombic
            if iframe % stride == 0:
                if names is None and first_names:
                    names = [(l[5:10].decode().strip(), l[10:15].decode().strip())
                             for l in raw]
                buf = b"".join(l[20:44] for l in raw)
                xyz = np.frombuffer(buf, dtype="S8").astype(float).reshape(n, 3)
                m = tre.search(title.decode())
                t = float(m.group(1)) if m else float(iframe)
                yield t, box, xyz, names
            iframe += 1


def water_indices(names):
    """Atom indices of OW, HW1, HW2 for every SOL molecule (from 1st frame)."""
    ow = [k for k, (r, a) in enumerate(names) if r == "SOL" and a == "OW"]
    h1 = [k for k, (r, a) in enumerate(names) if r == "SOL" and a == "HW1"]
    h2 = [k for k, (r, a) in enumerate(names) if r == "SOL" and a == "HW2"]
    assert len(ow) == len(h1) == len(h2) and len(ow) > 0, "water atoms not found"
    return np.array(ow), np.array(h1), np.array(h2)


# ----------------------------------------------------------------------------
# 2. Core: per-molecule F3 and F4
# ----------------------------------------------------------------------------
def mic(d, box):
    """Minimum-image convention for an orthorhombic box."""
    return d - box * np.round(d / box)


def neighbour_pairs(O, box, rc):
    """Directed neighbour list (i -> j, r_ij < rc) under PBC, sorted by i."""
    tree = cKDTree(O % box, boxsize=box)
    p = tree.query_pairs(rc, output_type="ndarray")
    i = np.concatenate([p[:, 0], p[:, 1]])
    j = np.concatenate([p[:, 1], p[:, 0]])
    o = np.argsort(i, kind="stable")
    return i[o], j[o]


def compute_F3_F4(O, H1, H2, box, rc=0.35):
    """
    O, H1, H2 : (N,3) arrays (nm) of water oxygen / hydrogen positions
    box       : (3,) orthorhombic box lengths (nm)
    rc        : first-solvation-shell cut-off for r_OO (nm); 0.35 nm is the
                first minimum of g_OO for TIP4P-type water.
    returns   : F3_i (N,), F4_i (N,)   (NaN when a molecule has too few
                neighbours: <2 for F3, <1 for F4)
    """
    N = len(O)
    i, j = neighbour_pairs(O, box, rc)
    cnt = np.bincount(i, minlength=N)

    # ---------------- F3: all O_j - O_i - O_k angles ----------------------
    start = np.cumsum(cnt) - cnt
    slot = np.arange(len(i)) - start[i]
    nmax = cnt.max()
    nbr = -np.ones((N, nmax), dtype=int)
    nbr[i, slot] = j
    valid = nbr >= 0
    d = mic(O[np.where(valid, nbr, 0)] - O[:, None, :], box)      # (N,nmax,3)
    u = d / np.linalg.norm(d, axis=2, keepdims=True)
    cos = np.einsum("nad,nbd->nab", u, u)                          # cos(theta_jik)
    f3 = (np.abs(cos) * cos + COS_TETRA ** 2) ** 2
    pm = valid[:, :, None] & valid[:, None, :]
    pm &= np.triu(np.ones((nmax, nmax), bool), k=1)[None]          # j<k, once
    npair = pm.sum(axis=(1, 2))
    F3 = np.full(N, np.nan)
    ok = npair > 0
    F3[ok] = (f3 * pm).sum(axis=(1, 2))[ok] / npair[ok]

    # ---------------- F4: H-O...O-H torsion for every O_i -> O_j ----------
    b = mic(O[j] - O[i], box)                                       # O_i -> O_j
    hi = np.stack([mic(H1[i] - O[i], box), mic(H2[i] - O[i], box)], 1)  # (P,2,3)
    hj = np.stack([mic(H1[j] - O[j], box), mic(H2[j] - O[j], box)], 1)
    # outer-most H of i = the one farther from O_j ; of j = farther from O_i
    di = np.linalg.norm(hi - b[:, None, :], axis=2)
    dj = np.linalg.norm(hj + b[:, None, :], axis=2)
    Hi = hi[np.arange(len(i)), np.argmax(di, axis=1)]
    Hj = hj[np.arange(len(i)), np.argmax(dj, axis=1)]
    a = -Hi                                  # O_i - H_i
    n1 = np.cross(a, b)                      # normal of plane (H_i, O_i, O_j)
    n2 = np.cross(b, Hj)                     # normal of plane (O_i, O_j, H_j)
    cphi = np.einsum("pd,pd->p", n1, n2) / (
        np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1))
    cos3phi = 4 * cphi ** 3 - 3 * cphi       # cos(3 phi) from cos(phi)
    F4 = np.full(N, np.nan)
    ok = cnt > 0
    F4[ok] = np.bincount(i, weights=cos3phi, minlength=N)[ok] / cnt[ok]
    return F3, F4


# ----------------------------------------------------------------------------
# 3. Self-tests on idealised geometries
# ----------------------------------------------------------------------------
def selftest():
    box = np.array([10.0, 10.0, 10.0])
    # (a) perfect tetrahedron: central O + 4 neighbours -> F3 = 0
    th = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    th = 0.27 * th / np.linalg.norm(th, axis=1)[:, None]
    O = np.vstack([[5, 5, 5], 5 + th])
    Hd = O + np.array([0.05, 0.05, 0.0])        # dummy hydrogens
    F3, _ = compute_F3_F4(O, Hd, Hd + 0.01, box)
    print("tetrahedron   F3(centre) = %.2e  (expect 0)" % F3[0])
    assert abs(F3[0]) < 1e-3          # cos(109.47) vs exact -1/3 -> ~1e-6
    # (b) planar 90-degree geometry -> F3 = cos^4(109.47)... check one value
    O = np.array([[5, 5, 5], [5.27, 5, 5], [5, 5.27, 5]], float)
    F3, _ = compute_F3_F4(O, O + [0, 0, 0.1], O + [0, 0.0, -0.1], box)
    print("90 deg trimer F3(centre) = %.4f  (expect %.4f)" % (F3[0], COS_TETRA ** 4))
    assert abs(F3[0] - COS_TETRA ** 4) < 1e-6
    # (c) dimer with prescribed torsion: eclipsed (0) -> +1, staggered (60) -> -1
    for phi_deg, expect in [(0, 1.0), (60, -1.0), (90, 0.0), (180, -1.0 + 0)]:
        phi = np.radians(phi_deg)
        O = np.array([[5, 5, 5], [5.28, 5, 5]], float)
        # outer H of molecule 0 lies away from O1, H of molecule 1 away from O0
        Hout0 = O[0] + 0.1 * np.array([-0.5, np.cos(0), np.sin(0)])
        Hout1 = O[1] + 0.1 * np.array([+0.5, np.cos(phi), np.sin(phi)])
        Hin0 = O[0] + 0.1 * np.array([+0.5, -1, 0])     # inner H (pointing to O1)
        Hin1 = O[1] + 0.1 * np.array([-0.5, -1, 0])
        _, F4 = compute_F3_F4(O, np.array([Hout0, Hout1]), np.array([Hin0, Hin1]), box)
        print("torsion %3d deg  F4 = %+.3f  (expect cos 3phi = %+.3f)"
              % (phi_deg, F4[0], np.cos(3 * phi)))
        assert abs(F4[0] - np.cos(3 * phi)) < 1e-6
    print("self-test passed")


# ----------------------------------------------------------------------------
# 4. Driver
# ----------------------------------------------------------------------------
def run(args):
    os.makedirs(args.out, exist_ok=True)
    rows, xt3, xt4, xtn = [], [], [], []
    ow = h1 = h2 = None
    all_F3, all_F4 = {}, {}
    t0 = time.time()
    for k, (t, box, xyz, names) in enumerate(iter_gro(args.gro, args.stride)):
        if ow is None:
            ow, h1, h2 = water_indices(names)
            nb = int(np.ceil(box[0] / args.binwidth))
            print("waters: %d | box %.3f x %.3f x %.3f nm | %d x-bins"
                  % (len(ow), *box, nb))
        F3, F4 = compute_F3_F4(xyz[ow], xyz[h1], xyz[h2], box, args.rc)
        rows.append((t, np.nanmean(F3), np.nanmean(F4), len(ow), *box))
        # x-profile (x = long axis); wrap to [0, Lx)
        x = xyz[ow][:, 0] % box[0]
        ib = np.minimum((x / box[0] * nb).astype(int), nb - 1)
        n = np.bincount(ib, minlength=nb).astype(float)
        s3 = np.bincount(ib[~np.isnan(F3)], weights=F3[~np.isnan(F3)], minlength=nb)
        s4 = np.bincount(ib[~np.isnan(F4)], weights=F4[~np.isnan(F4)], minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            xt3.append(s3 / n); xt4.append(s4 / n); xtn.append(n)
        if k == 0 or True:      # keep first / last frame distributions
            if k == 0:
                all_F3["first"], all_F4["first"] = F3, F4
            all_F3["last"], all_F4["last"] = F3, F4
        if k % 20 == 0:
            print("frame %4d  t=%9.1f ps  <F3>=%.4f  <F4>=%+.4f  (%.0fs)"
                  % (k, t, rows[-1][1], rows[-1][2], time.time() - t0), flush=True)

    rows = np.array(rows)
    np.savetxt(os.path.join(args.out, "F3_F4_vs_time.csv"), rows, delimiter=",",
               header="time_ps,F3_mean,F4_mean,n_water,Lx_nm,Ly_nm,Lz_nm", comments="")
    xt3, xt4, xtn = map(np.array, (xt3, xt4, xtn))
    Lx = rows[:, 4].mean()
    xc = (np.arange(nb) + 0.5) * Lx / nb
    with np.errstate(invalid="ignore"):
        p3 = np.nansum(xt3 * xtn, 0) / xtn.sum(0)
        p4 = np.nansum(xt4 * xtn, 0) / xtn.sum(0)
    np.savetxt(os.path.join(args.out, "F3_F4_profile_x.csv"),
               np.c_[xc, p3, p4, xtn.mean(0)], delimiter=",",
               header="x_nm,F3,F4,mean_waters_per_bin", comments="")
    np.savez(os.path.join(args.out, "F3_F4_xt.npz"), time_ps=rows[:, 0], x_nm=xc,
             F3=xt3, F4=xt4, n=xtn)
    plot(args, rows, xc, xt3, xt4, p3, p4, all_F3, all_F4)
    print("done in %.0f s -> %s" % (time.time() - t0, args.out))


def plot(args, rows, xc, xt3, xt4, p3, p4, all_F3, all_F4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_ns = rows[:, 0] / 1000.0
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].plot(t_ns, rows[:, 1], "C0"); ax[0, 0].set_ylabel("<F3>")
    a2 = ax[0, 0].twinx(); a2.plot(t_ns, rows[:, 2], "C3"); a2.set_ylabel("<F4>", color="C3")
    ax[0, 0].set_xlabel("time (ns)"); ax[0, 0].set_title("Box-averaged F3 (blue), F4 (red)")
    ax[0, 1].plot(xc, p3, "C0-o", ms=3); ax[0, 1].set_xlabel("x (nm)")
    ax[0, 1].set_ylabel("F3", color="C0"); ax[0, 1].set_title("Time-averaged profile along x")
    b2 = ax[0, 1].twinx(); b2.plot(xc, p4, "C3-o", ms=3); b2.set_ylabel("F4", color="C3")
    for a, M, lab, cm in [(ax[0, 2], xt3, "F3(x,t)", "viridis_r"),
                          (ax[1, 0], xt4, "F4(x,t)", "viridis")]:
        im = a.imshow(M.T, origin="lower", aspect="auto", cmap=cm,
                      extent=[t_ns[0], t_ns[-1], 0, xc[-1] + (xc[1] - xc[0]) / 2])
        a.set_xlabel("time (ns)"); a.set_ylabel("x (nm)"); a.set_title(lab)
        plt.colorbar(im, ax=a)
    for a, d, lab in [(ax[1, 1], all_F3, "F3"), (ax[1, 2], all_F4, "F4")]:
        for key, c in [("first", "C0"), ("last", "C3")]:
            v = d[key][~np.isnan(d[key])]
            a.hist(v, bins=60, alpha=0.5, color=c, density=True, label=key + " frame")
        a.set_xlabel(lab + " per molecule"); a.set_ylabel("probability density")
        a.set_title("Distribution of " + lab); a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "F3_F4_summary.png"), dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gro", nargs="?", help="multi-frame GROMACS .gro trajectory")
    ap.add_argument("--rc", type=float, default=0.35, help="O-O cut-off, nm (0.35)")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--binwidth", type=float, default=0.5, help="x-bin width, nm")
    ap.add_argument("--out", default="results_F3_F4")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.gro:
        run(a)
    else:
        ap.print_help()
        sys.exit(1)