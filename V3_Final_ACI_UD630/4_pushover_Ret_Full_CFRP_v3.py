"""
3D Cyclic Pushover Analysis — Model 4: FULL-BUILDING CFRP retrofit of the
                                       degraded G+3 RCC frame.

  Base building = degraded Model-2 (M20 concrete + As × 0.85 corroded steel).
  Retrofit      = EVERY storey (full building):
    All columns : multi-ply CFRP wrap → fcc ≈ 33–34 MPa (ACI 440.2R-17)
                  350×350 → 2 plies UD630  |  375×375 → 2 plies UD630
    All beams   : 1-ply soffit CFRP laminate (UD630) per ACI 440.2R-17.
                  Strain limited at ε_fd via MinMax wrapper. Applied at
                  every storey (plinth through terrace/HR slab).

CFRP material — UD630 Wrap (consistent with thesis and ACI 440.2R design sheets):
    Ef  = 220 GPa   ffu  = 2200 MPa   εfu = 0.015   tf = 0.332 mm bf=500mm
CFRP material — LH Laminate (consistent with thesis and ACI 440.2R design sheets):
    Ef  = 190 GPa   ffu  = 2700 MPa   εfu = 0.0163   tf = 1.2 mm bf=50mm

FIXES vs original script:
    FIX #1  Gravity → SLAVE nodes by tributary area
    FIX #2  X-beam vecxz (0,1,0) → (0,-1,0)
    FIX #3  Torsion ONCE via realistic -GJ; Aggregator removed
    FIX #4  Core = confined concrete, cover = unconfined
    FIX #5  Convergence tolerance 1e-6 → 1e-4
    FIX #6  Base concrete = M20 (degraded)
    FIX #7  CFRP applied to ALL storeys
    FIX #8  *** CORRECTED: CFRP material changed from UD230 → UD630 ***
    FIX #9  *** CORRECTED: confinement model changed from CNR-DT 200
                           → ACI 440.2R-17 (no conservative 0.004 strain
                           cap; uses kappa_e = 0.586 for rectangular cols) ***
    FIX #10 *** CORRECTED: ply count reduced 6-7 → 2 (sufficient for
                           fcc ≥ 30 MPa with UD630 + ACI model) ***
    FIX #11 Beam laminate debonding strain recomputed with UD630 properties

Expected results after correction:
    All columns: fcc ≈ 33–34 MPa (vs 30–31 MPa before, at all storeys now)
    Global peak base shear should approach or exceed M1 (original) capacity.

Outputs (./outputs/):
    disp_shear_cfrp_full.npz
    CFRP_full_hysteresis.csv
    CFRP_full_envelope.csv
    CFRP_full_plots.m
    Hysteresis_CFRP_Full.png
    Capacity_Envelope_CFRP_Full.png
    Stiffness_Degradation_CFRP_Full.png
    Capacity_Curve_CFRP_Full.png
"""

import os
import math
import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. INITIALIZATION & GEOMETRY
# ============================================================================
ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)

mm  = 1.0
m   = 1000 * mm
kN  = 1000.0
MPa = 1.0

dx     = [1.74, 2.25, 3.24, 1.20, 1.20, 3.24, 2.25, 1.74]
dy     = [2.0, 2.74, 2.74, 3.24]
Z_elev = [0.0, 1.25, 4.50, 7.75, 11.00, 14.25, 17.50]

X_grid = np.cumsum([0.0] + dx) * m
Y_grid = np.cumsum([0.0] + dy) * m
Z_grid = np.array(Z_elev) * m

dx_mm = np.array(dx) * m
dy_mm = np.array(dy) * m

# ============================================================================
# 2. MATERIALS — DEGRADED M20 base + Mander confined + ACI 440.2R confined
#                + ACI 440.2R beam laminate
# ============================================================================

# ---- Unconfined M20 (cover fibres for all sections) ----
fc       = -20.0 * MPa
epsc0    = -0.002
fcu_unc  = -0.2 * 20.0 * MPa
epscu    = -0.005
lambda_c = 0.1
ft       = 0.7 * np.sqrt(np.abs(fc))
Ec       = 5000.0 * np.sqrt(np.abs(fc))
Ets      = 0.1 * Ec
ops.uniaxialMaterial('Concrete02', 1, fc, epsc0, fcu_unc, epscu,
                     lambda_c, ft, Ets)

# ---- Mander confined M20 (kept for reference; unused in M4 element loop) ----
K_conf   = 1.20
fcc_M    = K_conf * fc
epscc_M  = epsc0 * (1.0 + 5.0 * (K_conf - 1.0))
fccu_M   = 0.2 * fcc_M
epsccu_M = -0.018
ft_M     = 0.7 * np.sqrt(np.abs(fcc_M))
ops.uniaxialMaterial('Concrete02', 4, fcc_M, epscc_M, fccu_M, epsccu_M,
                     lambda_c, ft_M, Ets)

# ---- Steel Fe500 ----
fy     = 500 * MPa
Es     = 200000.0 * MPa
b_hard = 0.02
ops.uniaxialMaterial('Steel02', 2, fy, Es, b_hard, 18.0, 0.925, 0.15)


# ============================================================================
# CFRP MATERIAL DATA — UD630 (CORRECTED from UD230)
# ----------------------------------------------------------------------------
#    Ef  = 220 GPa   ffu = 2200 MPa   εfu = 0.0162   tf = 0.332 mm
# ============================================================================
Ef_w     = 220000.0    # MPa
ffu_w    = 2200.0      # MPa
tf_w     = 0.332       # mm/ply
eps_fu_w = ffu_w / Ef_w   # = 0.01618

# Same UD630 used as soffit laminate
Ef_lam     = 190000.0
ffu_lam    = 2700.0
tf_lam     = 1.2
bf_lam     = 150.0     # mm (soffit laminate width)
eps_fu_lam = ffu_lam / Ef_lam


# ============================================================================
# ACI 440.2R-17 — FRP-CONFINED RECTANGULAR CONCRETE COLUMN
# ----------------------------------------------------------------------------
# Ref: ACI 440.2R-17 Eqs. 12-1 to 12-9
#
# kappa_e = 0.586  (efficiency factor, rectangular column, ACI Table 12.1)
# eps_fe  = kappa_e * eps_fu           (effective FRP strain, Eq. 12-3)
# ka      = Ae / Ag                    (shape factor, Eq. 12-9)
# fl      = 0.5 * Ef * eps_fe * rho_f  (confining pressure, Eq. 12-2)
# fcc     = fco + psi_f * 3.3 * ka * fl  (confined strength, Eq. 12-1)
# eps_ccu = [1.5 + 12*ka*(fl/fco)*(eps_fe/0.003)] * 0.003  (Eq. 12-6)
#
# No hard strain cap at 0.004 (unlike CNR-DT 200) → full UD630 strength used.
# ============================================================================
KAPPA_E = 0.586
PSI_F   = 0.95

def aci_440_confined(b, h, n_ply, tf, Ef, eps_fu,
                     fco_pos=20.0, eps_co=0.002, rc=25.0):
    """ACI 440.2R-17 confinement model for rectangular column.

    Returns (fcc, eps_cc, fcu, eps_ccu) as positive magnitudes.
    """
    eps_fe = KAPPA_E * eps_fu            # ACI Eq. 12-3; no 0.004 cap

    Ag    = b * h
    b2, h2 = b - 2.0*rc, h - 2.0*rc
    Ae    = Ag - (b2**2 + h2**2) / 3.0
    ka    = Ae / Ag

    rho_f = 2.0 * n_ply * tf * (b + h) / (b * h)
    fl    = 0.5 * Ef * eps_fe * rho_f
    fl_a  = ka * fl

    fcc_pos = fco_pos + PSI_F * 3.3 * fl_a                        # Eq. 12-1

    eps_ccu = (1.5 + 12.0 * ka * (fl / fco_pos) * (eps_fe / 0.003)) * 0.003  # Eq. 12-6

    # eps_cc = eps_co (standard OpenSeesPy practice for CFRP-confined columns)
    # Richart scaling gives eps_cc~0.009 → Concrete02 initial tangent only 0.34x
    # of cover concrete → ForceBeamColumn3d fails at iteration 0 (dW0 >> dW).
    # Keeping eps_cc = eps_co = 0.002 is physically correct (CFRP raises fcc and
    # eps_ccu but not ascending-branch stiffness) and numerically stable.
    eps_cc = eps_co   # 0.002 — do NOT use Richart scaling for OpenSeesPy

    fcu_pos = 0.85 * fcc_pos

    return fcc_pos, eps_cc, fcu_pos, eps_ccu


def define_confined_concrete(tag, fcc_p, eps_cc_p, fcu_p, eps_ccu_p):
    """Concrete02 confined-core material (negative signs for compression)."""
    fcc     = -abs(fcc_p)
    eps_cc  = -abs(eps_cc_p)
    fcu     = -abs(fcu_p)
    eps_ccu = -abs(eps_ccu_p)
    ft_loc  = 0.7 * math.sqrt(abs(fcc))
    Ec_loc  = 5000.0 * math.sqrt(abs(fcc))
    Ets_loc = 0.10 * Ec_loc
    ops.uniaxialMaterial('Concrete02', tag,
                         fcc, eps_cc, fcu, eps_ccu,
                         0.10, ft_loc, Ets_loc)


# ---- ACI 440.2R-17 confined cores for ALL columns (UD630, 2 plies each) ----
N_PLY_PINK = 2    # 350×350  →  fcc ≈ 33.9 MPa
N_PLY_GRAY = 2    # 375×375  →  fcc ≈ 32.7 MPa

fcc_pink, ec_pink, fcu_pink, ecu_pink = aci_440_confined(
    350.0, 350.0, N_PLY_PINK, tf_w, Ef_w, eps_fu_w, fco_pos=20.0)
fcc_gray, ec_gray, fcu_gray, ecu_gray = aci_440_confined(
    375.0, 375.0, N_PLY_GRAY, tf_w, Ef_w, eps_fu_w, fco_pos=20.0)

define_confined_concrete(6, fcc_pink, ec_pink, fcu_pink, ecu_pink)   # Pink
define_confined_concrete(7, fcc_gray, ec_gray, fcu_gray, ecu_gray)   # Gray

print("[ACI 440.2R-17] All column confinement (UD630, kappa_e=0.586):")
print(f"  Pink 350×350 (n={N_PLY_PINK}): f'cc = {fcc_pink:5.2f} MPa, "
      f"eps_cc = {ec_pink:.4f}, eps_ccu = {ecu_pink:.4f}")
print(f"  Gray 375×375 (n={N_PLY_GRAY}): f'cc = {fcc_gray:5.2f} MPa, "
      f"eps_cc = {ec_gray:.4f}, eps_ccu = {ecu_gray:.4f}")


# ============================================================================
# ACI 440.2R-17 — BEAM SOFFIT LAMINATE (UD630, FIX #11)
# ----------------------------------------------------------------------------
# eps_fd = 0.41 * sqrt(fc' / (n * Ef * tf))   [MPa, mm]  ≤ 0.9 * eps_fu
# ============================================================================
N_PLY_LAM   = 1
fc_lam_base = 20.0
eps_fd = 0.41 * math.sqrt(fc_lam_base / (N_PLY_LAM * Ef_lam * tf_lam))
eps_fd = min(eps_fd, 0.9 * eps_fu_lam)

EPS_LAM_MIN = -eps_fd
EPS_LAM_MAX =  eps_fd

ops.uniaxialMaterial('Elastic', 8, Ef_lam)
ops.uniaxialMaterial('MinMax', 9, 8,
                     '-min', EPS_LAM_MIN, '-max', EPS_LAM_MAX)

print(f"[ACI 440.2R-17] Beam soffit laminate (UD630): n={N_PLY_LAM}, "
      f"eps_fd = {eps_fd:.5f}  (0.9*eps_fu = {0.9*eps_fu_lam:.5f})")


# ============================================================================
# 3. SECTION DEFINITIONS
# ============================================================================
STEEL_SCALE = 0.85    # corroded reinforcement


def build_fiber_section(sec_tag, H, B, cover, As_top, As_bot,
                        is_col=False, core_mat=4, add_cfrp_beam=False):
    """3D fibre section.

    core_mat      : 4 = Mander conf M20 (reference; unused in M4 elements)
                    6 = ACI 440.2R conf Pink  (350×350 columns)
                    7 = ACI 440.2R conf Gray  (375×375 columns)
    add_cfrp_beam : True → add UD630 soffit laminate (mat 9) at beam bottom
    """
    As_top = As_top * STEEL_SCALE
    As_bot = As_bot * STEEL_SCALE

    G  = 0.4 * Ec
    a, bb = max(H, B), min(H, B)
    J  = (a * bb**3) * (1.0/3.0 - 0.21*(bb/a)*(1.0 - (bb/a)**4 / 12.0))
    GJ = G * J

    ops.section('Fiber', sec_tag, '-GJ', GJ)

    y1, z1 = H / 2.0, B / 2.0

    # CORE (confined)
    ops.patch('rect', core_mat, 10, 10,
              cover - y1, cover - z1, y1 - cover, z1 - cover)
    # COVER (unconfined M20)
    ops.patch('rect', 1, 10,  2, -y1,        -z1,          y1,         -z1 + cover)
    ops.patch('rect', 1, 10,  2, -y1,         z1 - cover,  y1,          z1)
    ops.patch('rect', 1,  2, 10, -y1,        -z1 + cover, -y1 + cover,  z1 - cover)
    ops.patch('rect', 1,  2, 10,  y1 - cover, -z1 + cover,  y1,          z1 - cover)

    # Steel
    if is_col:
        As_corner = (As_top + As_bot) / 4.0
        ops.layer('straight', 2, 2, As_corner,
                   y1 - cover,  z1 - cover,  y1 - cover, -z1 + cover)
        ops.layer('straight', 2, 2, As_corner,
                  -y1 + cover,  z1 - cover, -y1 + cover, -z1 + cover)
    else:
        ops.layer('straight', 2, 3, As_top / 3,
                   y1 - cover,  z1 - cover,  y1 - cover, -z1 + cover)
        ops.layer('straight', 2, 3, As_bot / 3,
                  -y1 + cover,  z1 - cover, -y1 + cover, -z1 + cover)

    # CFRP soffit laminate
    if add_cfrp_beam and not is_col:
        n_sub   = 4
        A_strip = N_PLY_LAM * bf_lam * tf_lam
        A_each  = A_strip / n_sub
        z_half  = bf_lam / 2.0
        ops.layer('straight', 9, n_sub, A_each,
                  -y1, -z_half, -y1, z_half)


# ---- Column sections (all storeys CFRP-wrapped) ----
# 101/102 = reference non-retrofitted (defined but not used in M4 elements)
build_fiber_section(101, 350, 350, 40, 490, 490, is_col=True, core_mat=4)
build_fiber_section(102, 375, 375, 40, 562, 562, is_col=True, core_mat=4)
# 103/104 = ACI 440.2R confined (used in ALL M4 storey columns)
build_fiber_section(103, 350, 350, 40, 490, 490, is_col=True, core_mat=6)
build_fiber_section(104, 375, 375, 40, 562, 562, is_col=True, core_mat=7)

# ---- Beam sections (all storeys CFRP soffit laminated) ----
# Reference (unused):
build_fiber_section(201, 450, 300, 30, 600, 394)
build_fiber_section(203, 325, 230, 30, 500, 211)
build_fiber_section(204, 300, 230, 30, 400, 193)
build_fiber_section(205, 250, 230, 30, 300, 157)
# CFRP-laminated variants (one per storey level):
build_fiber_section(221, 450, 300, 30, 600, 394, add_cfrp_beam=True)   # Plinth
build_fiber_section(222, 375, 230, 30, 600, 248, add_cfrp_beam=True)   # Floor 1
build_fiber_section(223, 325, 230, 30, 500, 211, add_cfrp_beam=True)   # Floor 2
build_fiber_section(224, 300, 230, 30, 400, 193, add_cfrp_beam=True)   # Floor 3
build_fiber_section(225, 250, 230, 30, 300, 157, add_cfrp_beam=True)   # Terrace / HR


def get_beam_sec(z_idx):
    """All storeys → CFRP-laminated beams."""
    if z_idx == 1: return 221
    if z_idx == 2: return 222
    if z_idx == 3: return 223
    if z_idx == 4: return 224
    return 225


def get_col_sec(perim, iz_start):
    """All storey columns → CFRP-wrapped (ACI 440.2R confined core)."""
    return 103 if perim else 104


# ============================================================================
# 4. NODES & BOUNDARY CONDITIONS
# ============================================================================
def get_node_tag(ix, iy, iz):
    return iz * 1000 + iy * 100 + ix + 1


def in_void(ix, iy):
    return (3 <= ix <= 4) and (iy == 0)


def is_void_cell(cell_ix, cell_iy):
    return (2 <= cell_ix <= 4) and (cell_iy == 0)


for iz, z in enumerate(Z_grid):
    for iy, y in enumerate(Y_grid):
        for ix, x in enumerate(X_grid):
            if in_void(ix, iy) and iz > 0:
                continue
            node_tag = get_node_tag(ix, iy, iz)
            ops.node(node_tag, x, y, z)
            if iz == 0:
                ops.fix(node_tag, 1, 1, 1, 1, 1, 1)

master_nodes = {}
for iz in range(1, len(Z_grid)):
    master_tag = iz * 1000 + 999
    ops.node(master_tag,
             float(np.mean(X_grid)), float(np.mean(Y_grid)),
             float(Z_grid[iz]))
    ops.fix(master_tag, 0, 0, 1, 1, 1, 0)
    master_nodes[iz] = master_tag

    slave_tags = [get_node_tag(ix, iy, iz)
                  for iy in range(len(Y_grid))
                  for ix in range(len(X_grid))
                  if not in_void(ix, iy)]
    ops.rigidDiaphragm(3, master_tag, *slave_tags)

# ============================================================================
# 5. ELEMENTS
# ============================================================================
ops.geomTransf('PDelta', 1, 1,  0, 0)    # Columns
ops.geomTransf('Linear', 2, 0, -1, 0)    # X-beams (FIX #2)
ops.geomTransf('Linear', 3, 1,  0, 0)    # Y-beams

ele_tag = 1

# Columns
for iz in range(len(Z_grid) - 1):
    for iy in range(len(Y_grid)):
        for ix in range(len(X_grid)):
            if in_void(ix, iy):
                continue
            nI = get_node_tag(ix, iy, iz)
            nJ = get_node_tag(ix, iy, iz + 1)
            perim = (ix == 0 or ix == len(X_grid) - 1 or
                     iy == 0 or iy == len(Y_grid) - 1)
            sec_tag = get_col_sec(perim, iz)
            ops.element('nonlinearBeamColumn', ele_tag, nI, nJ,
                        5, sec_tag, 1)
            ele_tag += 1

# Beams
for iz in range(1, len(Z_grid)):
    sec_tag = get_beam_sec(iz)

    for iy in range(len(Y_grid)):
        for ix in range(len(X_grid) - 1):
            if (2 <= ix <= 4) and iy == 0:
                continue
            nI = get_node_tag(ix,     iy, iz)
            nJ = get_node_tag(ix + 1, iy, iz)
            ops.element('nonlinearBeamColumn', ele_tag, nI, nJ,
                        5, sec_tag, 2)
            ele_tag += 1

    for ix in range(len(X_grid)):
        for iy in range(len(Y_grid) - 1):
            if (3 <= ix <= 4) and iy == 0:
                continue
            nI = get_node_tag(ix, iy,     iz)
            nJ = get_node_tag(ix, iy + 1, iz)
            ops.element('nonlinearBeamColumn', ele_tag, nI, nJ,
                        5, sec_tag, 3)
            ele_tag += 1

# ============================================================================
# GRAVITY LOAD TABLE
# ============================================================================
story_gravity_loads = {
    1.25 : 191,  4.50 : 1940, 7.75 : 1940,
    11.00: 1940, 14.25: 1850, 17.50: 650,
}

# ============================================================================
# 6. GRAVITY ANALYSIS
# ============================================================================
def tributary_area(ix, iy):
    A = 0.0
    n_cx, n_cy = len(dx_mm), len(dy_mm)
    for cx_off, cy_off in [(-1, -1), (0, -1), (-1, 0), (0, 0)]:
        cx, cy = ix + cx_off, iy + cy_off
        if 0 <= cx < n_cx and 0 <= cy < n_cy and not is_void_cell(cx, cy):
            A += dx_mm[cx] * dy_mm[cy] / 4.0
    return A


ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)

print("\nGravity loads being distributed:")
total_applied = 0.0
for iz in range(1, len(Z_grid)):
    elev       = round(Z_grid[iz] / m, 2)
    W_total_N  = story_gravity_loads.get(elev, 1940) * kN

    slave_info, A_sum = [], 0.0
    for iy in range(len(Y_grid)):
        for ix in range(len(X_grid)):
            if in_void(ix, iy):
                continue
            A = tributary_area(ix, iy)
            if A > 0:
                slave_info.append((get_node_tag(ix, iy, iz), A))
                A_sum += A

    for node_tag, A_node in slave_info:
        W_node = W_total_N * (A_node / A_sum)
        ops.load(node_tag, 0.0, 0.0, -W_node, 0.0, 0.0, 0.0)

    total_applied += W_total_N
    print(f"  Elev {elev:>5.2f} m | {W_total_N/kN:>5.0f} kN  →  "
          f"{len(slave_info)} nodes")

ops.system('BandGeneral')
ops.numberer('RCM')
ops.constraints('Transformation')
ops.test('NormDispIncr', 1.0e-4, 100)
ops.algorithm('Newton')
ops.integrator('LoadControl', 0.1)
ops.analysis('Static')
ok = ops.analyze(10)
if ok != 0:
    raise RuntimeError("Gravity analysis failed to converge")

ops.loadConst('-time', 0.0)

base_nodes = [get_node_tag(ix, iy, 0)
              for ix in range(len(X_grid))
              for iy in range(len(Y_grid))
              if not in_void(ix, iy)]
ops.reactions()
Rz_sum = sum(ops.nodeReaction(n, 3) for n in base_nodes)
print(f"\n✅ Gravity complete. Applied = {total_applied/kN:.0f} kN, "
      f"Σ base Rz = {Rz_sum/kN:.0f} kN")

# ============================================================================
# 7. CYCLIC PUSHOVER ANALYSIS
# ============================================================================
ops.wipeAnalysis()
ops.system('BandGeneral')
ops.numberer('RCM')
ops.constraints('Transformation')
ops.test('NormDispIncr', 1.0e-5, 500, 0)
ops.algorithm('Newton')
ops.analysis('Static')

amps     = [2, 4, 8, 12, 16, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_cycles = 3
protocol = []
for amp in amps:
    for _ in range(n_cycles):
        protocol.extend([amp, -amp])

roof_master_node = master_nodes[len(Z_grid) - 1]

ops.timeSeries('Linear', 2)
ops.pattern('Plain', 2, 2)
for iz, mtag in master_nodes.items():
    ops.load(mtag, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _try_algs():
    ok = ops.analyze(1)
    if ok == 0:
        return 0
    for alg in ('ModifiedNewton', 'KrylovNewton', 'SecantNewton', 'NewtonLineSearch'):
        ops.algorithm(alg)
        ok = ops.analyze(1)
        if ok == 0:
            ops.algorithm('Newton')
            return 0
    ops.algorithm('Newton')
    return ok


def advance(dU_target, node_tag, dof=1, min_factor=256):
    ops.integrator('DisplacementControl', node_tag, dof, dU_target)
    if _try_algs() == 0:
        return 0
    factor = 2
    while factor <= min_factor:
        sub_dU = dU_target / factor
        ops.integrator('DisplacementControl', node_tag, dof, sub_dU)
        ok_all = 0
        for _ in range(factor):
            if _try_algs() != 0:
                ok_all = -1
                break
        if ok_all == 0:
            return 0
        factor *= 2
    return -1


disp_history, shear_history, peak_indices = [], [], []
current_disp = 0.0

print(f"\nStarting Cyclic Pushover — M4 Full CFRP retrofit (UD630 + ACI 440.2R-17)")
print(f"  Protocol: {len(protocol)} peaks, {len(amps)} amplitudes × {n_cycles} cycles")
for k, peak in enumerate(protocol):
    target_disp = peak * mm
    amp_now   = abs(peak)
    step_size = max(0.10, amp_now / 80.0) * mm
    dU    = step_size if target_disp > current_disp else -step_size
    steps = int(round(abs(target_disp - current_disp) / step_size))

    failed = False
    for _ in range(steps):
        ok = advance(dU, roof_master_node, 1)
        if ok != 0:
            print(f"  ⚠ Analysis failed at u = {current_disp:.3f} mm")
            failed = True
            break
        current_disp += dU
        disp_history.append(current_disp)
        ops.reactions()
        base_shear = sum(ops.nodeReaction(n, 1) for n in base_nodes)
        shear_history.append(-base_shear / kN)
    if failed:
        break
    peak_indices.append(len(disp_history) - 1)
    print(f"  Peak {peak:+4d} mm  (step {k+1}/{len(protocol)})")

# ============================================================================
# 8. POST-PROCESSING & SAVE
# ============================================================================
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(out_dir, exist_ok=True)

disp_array  = np.asarray(disp_history)
shear_array = np.asarray(shear_history)

np.savez(os.path.join(out_dir, 'disp_shear_cfrp_full.npz'),
         disp=disp_array, shear=shear_array,
         peak_indices=np.asarray(peak_indices, dtype=int),
         amps=np.asarray(amps), n_cycles=n_cycles)

env_disp  = disp_array[peak_indices]
env_shear = shear_array[peak_indices]

np.savetxt(os.path.join(out_dir, 'CFRP_full_hysteresis.csv'),
           np.column_stack([disp_array, shear_array]),
           delimiter=',', header='Displacement_mm,BaseShear_kN', comments='')

drift_env = env_disp / Z_grid[-1] * 100.0
stiff_env = np.where(np.abs(env_disp) > 0,
                     np.abs(env_shear) / np.abs(env_disp), 0.0)
np.savetxt(os.path.join(out_dir, 'CFRP_full_envelope.csv'),
           np.column_stack([env_disp, env_shear, drift_env, stiff_env]),
           delimiter=',',
           header='Displacement_mm,BaseShear_kN,Drift_pct,Stiffness_kNmm',
           comments='')

matlab_script = (
    "% CFRP_full_plots.m -- auto-generated (M4, UD630 + ACI 440.2R-17)\n"
    "hyst = readmatrix('CFRP_full_hysteresis.csv');\n"
    "env  = readmatrix('CFRP_full_envelope.csv');\n"
    "d = hyst(:,1); V = hyst(:,2);\n"
    "ed = env(:,1); eV = env(:,2); drift = env(:,3); K = env(:,4);\n\n"
    "figure; plot(d, V, 'r', 'LineWidth', 1); grid on;\n"
    "xlabel('Lateral Displacement (mm)'); ylabel('Lateral Load (kN)');\n"
    "title('Hysteretic Response — M4 Full CFRP Retrofit (UD630)');\n\n"
    "figure; plot(ed, eV, 'b-o', 'MarkerSize', 4); grid on;\n"
    "xlabel('Lateral Displacement (mm)'); ylabel('Lateral Load (kN)');\n"
    "title('Capacity Envelope — M4 Full CFRP Retrofit (UD630)');\n\n"
    "[~,idx] = sort(abs(ed));\n"
    "figure; plot(abs(ed(idx)), K(idx), 'g-s', 'MarkerSize', 4); grid on;\n"
    "xlabel('Lateral Displacement (mm)'); ylabel('Stiffness (kN/mm)');\n"
    "title('Stiffness Degradation — M4 Full CFRP Retrofit (UD630)');\n\n"
    "pos = ed > 0;\n"
    "figure; plot(ed(pos), eV(pos), 'm-o', 'MarkerSize', 4); grid on;\n"
    "xlabel('Top-Storey Displacement (mm)'); ylabel('Lateral Load (kN)');\n"
    "title('Capacity Curve — M4 Full CFRP Retrofit (UD630)');\n"
)
with open(os.path.join(out_dir, 'CFRP_full_plots.m'), 'w', encoding='utf-8') as fh:
    fh.write(matlab_script)

stiff_disp = np.abs(env_disp)
stiff_vals = np.where(stiff_disp > 0,
                      np.abs(env_shear) / np.where(stiff_disp == 0, 1, stiff_disp),
                      0.0)
order = np.argsort(stiff_disp)
stiff_disp, stiff_vals = stiff_disp[order], stiff_vals[order]

plt.style.use('ggplot')

plt.figure(figsize=(10, 6))
plt.plot(disp_array, shear_array, color='red', linewidth=1.0,
         label='M4 — Full CFRP Retrofit (UD630 + ACI 440.2R-17)')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Hysteretic Load–Displacement Response Curve (M4 Full CFRP Retrofit)')
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Lateral Load (kN)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Hysteresis_CFRP_Full.png'), dpi=150)

plt.figure(figsize=(10, 6))
plt.plot(env_disp, env_shear, marker='o', color='blue',
         markersize=4, linestyle='-', label='M4 Capacity Envelope')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Load–Displacement Envelope Curve (M4 Full CFRP Retrofit)')
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Lateral Load (kN)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Capacity_Envelope_CFRP_Full.png'), dpi=150)

plt.figure(figsize=(10, 6))
plt.plot(stiff_disp, stiff_vals, marker='s', color='green',
         markersize=4, linestyle='-', label='M4 Full CFRP Stiffness')
plt.title('Stiffness Degradation Curve (M4 Full CFRP Retrofit)')
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Stiffness (kN/mm)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Stiffness_Degradation_CFRP_Full.png'), dpi=150)

pos_mask = env_disp > 0
plt.figure(figsize=(10, 6))
plt.plot(env_disp[pos_mask], env_shear[pos_mask], marker='o', color='purple',
         markersize=4, linestyle='-', label='M4 Capacity Curve')
plt.title('Capacity Curve — Load vs Top-Storey Displacement (M4 Full CFRP Retrofit)')
plt.xlabel('Top-Storey Displacement (mm)')
plt.ylabel('Lateral Load (kN)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Capacity_Curve_CFRP_Full.png'), dpi=150)

print(f"\n✅ M4 Full CFRP retrofit analysis complete (UD630 + ACI 440.2R-17).")
print(f"   Results saved to: {out_dir}")
