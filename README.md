# CB4103 – Three-body (F3) and four-body (F4) order parameters

**Code:** `order_parameters_F3_F4.py`  (Python 3, numpy, scipy, matplotlib)
**Reference:** Rodger, Forester & Smith, *Fluid Phase Equilib.* 116, 326 (1996).
**Run:** `python order_parameters_F3_F4.py trj_HTR.gro` (≈15 s for all 317 frames);
`--selftest` checks the code on ideal geometries.

## Definitions implemented
* **F3 = ⟨[ |cosθ|·cosθ + cos²(109.47°) ]²⟩** – θ is the O_j–O_i–O_k angle formed by
  oxygen *i* and two oxygens in its first solvation shell. F3 = 0 for a perfect tetrahedral
  network and increases with disorder.
* **F4 = ⟨cos 3φ⟩** – φ is the H–O···O–H torsion of two neighbouring waters, using the
  outer-most hydrogen of each molecule (the H farther from the partner oxygen).
  F4 ≈ 1 for the eclipsed, planar-ring hydrate lattice, ≈ 0 for liquid water.

## Procedure
1. **Read the trajectory** frame by frame (streaming reader, one frame in memory; box and time
   taken from every frame). Water atoms OW, HW1, HW2 are located by name from the first frame;
   the TIP4P virtual site MW, CO2 and THF are ignored.
2. **Neighbour list** of water oxygens with a periodic cKDTree, r_OO < 0.35 nm
   (first minimum of g_OO; adjustable with `--rc`). All distance/angle vectors use the
   minimum-image convention.
3. **F3 per molecule:** for each oxygen *i*, build the unit vectors to its neighbours,
   compute cosθ for all neighbour pairs (j<k), evaluate (|cosθ|cosθ + cos²109.47°)² and average.
4. **F4 per molecule:** for every neighbour pair (i,j) pick the outer-most H on each molecule,
   compute the dihedral from the normals of planes (H_i,O_i,O_j) and (O_i,O_j,H_j),
   evaluate cos3φ = 4cos³φ − 3cosφ and average over all neighbours of *i*.
5. **Averaging:** per-molecule values → mean over all waters per frame (time series) and
   over x-bins of 0.5 nm (profile along the long box axis, which crosses the hydrate/solution
   interface). Per-frame x-profiles are kept to draw F3(x,t) and F4(x,t) maps.
6. **Validation:** perfect tetrahedron → F3 = 0; 90° trimer → F3 = cos⁴109.47°; ideal dimers with
   φ = 0°, 60°, 90°, 180° → F4 = +1, −1, 0, −1 (`--selftest`).

## Results (trj_HTR.gro: 4416 waters, 317 frames, 0–316 ns)
| region | F3 | F4 |
|---|---|---|
| hydrate slab (x ≈ 7–10 nm, first 10 ns) | ≈ 0.015 | ≈ 0.70 |
| liquid water (x > 13 nm) | ≈ 0.07 | ≈ 0.03 |

These match the bulk-hydrate (F4 ≈ 0.7, F3 ≈ 0.01) vs liquid (F4 ≈ 0) values of Rodger et al.
Over time the slab shrinks from both edges and is gone by ≈ 285 ns: the box-averaged
F4 falls from 0.33 to ≈ 0 and F3 rises from 0.044 to ≈ 0.075 (hydrate dissociation).

Output files are in `results/`: `F3_F4_vs_time.csv`, `F3_F4_profile_x.csv`, `F3_F4_xt.npz`, `F3_F4_summary.png`.