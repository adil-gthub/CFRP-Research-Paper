"""
3D Cyclic Pushover Analysis — bare RCC frame (G+3 + Plinth + Terrace + HR slab).

CORRECTED VERSION — fixes applied:
    FIX #1  Gravity loads now applied to SLAVE (structural) nodes by tributary
            area, instead of being absorbed by the master-node Z-fixity.
    FIX #2  X-beam geomTransf changed from (0, 1, 0) → (0, -1, 0) so that
            beam local-y axis = +Z (top steel actually at the top, matches
            Y-beams).
    FIX #3  Torsion defined ONCE via realistic -GJ on the Fiber section.
            Aggregator removed.  GJ now uses G ≈ 0.4·Ec and St-Venant J.
    FIX #4  Mander confined-concrete material added (mat 4) and used for the
            core patch; cover patches keep unconfined M30 (mat 1).  Comment
            mismatch (M20 vs M30) cleaned up.
    FIX #5  Convergence tolerance relaxed from 1e-6 → 1e-4 (still 0.1 µm —
            plenty tight) so the solver doesn't bounce around at large drifts.
    BONUS   Step size now adapts with amplitude — small at low drift for
            resolution, larger at high drift for speed.

Outputs:
    outputs/disp_shear_rcc.npz   (disp, shear, peak_indices, n_cycles, amps)
    outputs/Hysteresis_RCC.png
    outputs/Capacity_Envelope_RCC.png
    outputs/Stiffness_Degradation_RCC.png
"""

import os
import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. INITIALIZATION & GEOMETRY
# ============================================================================
ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)

# Units: N, mm, MPa  (consistent throughout)
mm  = 1.0
m   = 1000 * mm
kN  = 1000.0
MPa = 1.0

# Bay widths (m) and storey elevations (m)
dx     = [1.74, 2.25, 3.24, 1.20, 1.20, 3.24, 2.25, 1.74]
dy     = [2.0, 2.74, 2.74, 3.24]
Z_elev = [0.0, 1.25, 4.50, 7.75, 11.00, 14.25, 17.50]

X_grid = np.cumsum([0.0] + dx) * m
Y_grid = np.cumsum([0.0] + dy) * m
Z_grid = np.array(Z_elev) * m

dx_mm = np.array(dx) * m         # bay widths in mm (used for tributary areas)
dy_mm = np.array(dy) * m

# ============================================================================
# 2. MATERIALS  (FIX #4: add Mander confined concrete; M30 throughout)
# ============================================================================

# ---- Unconfined concrete M30 (Concrete02) — used in the COVER fibres ----
fc       = -30.0 * MPa            # peak compressive strength (negative)
epsc0    = -0.002                 # strain at peak
fcu      = -0.2 * 30.0 * MPa      # residual strength (20 % of fc)
epscu    = -0.005                 # ultimate strain
lambda_c = 0.1                    # unloading slope ratio
ft       = 0.7 * np.sqrt(np.abs(fc))   # tensile strength ≈ 3.83 MPa
Ec       = 5000.0 * np.sqrt(np.abs(fc))   # IS 456 secant modulus (≈ 27,386 MPa)
Ets      = 0.1 * Ec               # tension softening modulus
ops.uniaxialMaterial('Concrete02', 1, fc, epsc0, fcu, epscu,
                     lambda_c, ft, Ets)

# ---- Confined concrete (Mander model) — used in the CORE fibres ----
# Assumes ductile detailing: 8 mm stirrups @ 150 mm c/c with 4-leg ties.
# This gives a typical confinement effectiveness factor K ≈ 1.25 for M30.
# (If you later want to be precise per Mander 1988, plug in your actual
#  stirrup geometry; the formulae below already follow Mander's form.)
K_conf  = 1.25
fcc     = K_conf * fc                                    # ≈ −37.5 MPa
epscc   = epsc0 * (1.0 + 5.0 * (K_conf - 1.0))           # Mander eq. ≈ −0.0045
fccu    = 0.2 * fcc                                      # residual ≈ −7.5 MPa
epsccu  = -0.020                                         # ultimate (Mander 0.014–0.025)
ft_c    = 0.7 * np.sqrt(np.abs(fcc))
ops.uniaxialMaterial('Concrete02', 4, fcc, epscc, fccu, epsccu,
                     lambda_c, ft_c, Ets)

# ---- Steel Fe500 (Steel02) ----
fy     = 500 * MPa
Es     = 200000.0 * MPa
b_hard = 0.02
ops.uniaxialMaterial('Steel02', 2, fy, Es, b_hard, 18.0, 0.925, 0.15)

# NOTE: torsion material removed.  Section torsion now via -GJ flag (FIX #3).

# ============================================================================
# 3. SECTION DEFINITIONS  (FIX #3 + FIX #4)
# ============================================================================
def build_fiber_section(sec_tag, H, B, cover, As_top, As_bot, is_col=False):
    """Build a 3D fibre section.

    Core patch  → confined concrete (mat 4, Mander)
    Cover patch → unconfined concrete (mat 1, M30)
    Steel       → mat 2 (Fe500)
    Torsion     → ONE-shot via realistic -GJ on the Fiber section.

    H, B, cover, As_* all in mm / mm².
    """
    # --- Realistic St-Venant torsion stiffness for a solid rectangle ---
    G = 0.4 * Ec                       # concrete shear modulus
    a, bb = max(H, B), min(H, B)
    # Closed-form J for a rectangle (Roark / Timoshenko)
    J = (a * bb**3) * (1.0/3.0 - 0.21*(bb/a)*(1.0 - (bb/a)**4 / 12.0))
    GJ = G * J                         # e.g. 350×350 → GJ ≈ 5.7×10¹³ N·mm²

    ops.section('Fiber', sec_tag, '-GJ', GJ)

    y1, z1 = H / 2.0, B / 2.0

    # --- Concrete patches ---
    # CORE (mat 4 — confined)
    ops.patch('rect', 4, 10, 10, cover - y1, cover - z1,
              y1 - cover, z1 - cover)
    # COVER strips around the core (mat 1 — unconfined)
    ops.patch('rect', 1, 10,  2, -y1,        -z1,         y1,         -z1 + cover)
    ops.patch('rect', 1, 10,  2, -y1,         z1 - cover, y1,         z1)
    ops.patch('rect', 1,  2, 10, -y1,        -z1 + cover, -y1 + cover, z1 - cover)
    ops.patch('rect', 1,  2, 10,  y1 - cover, -z1 + cover, y1,         z1 - cover)

    # --- Reinforcement layers ---
    if is_col:
        # 4 corner bars, each with area = (top+bot)/4
        As_corner = (As_top + As_bot) / 4.0
        ops.layer('straight', 2, 2, As_corner,
                   y1 - cover, z1 - cover,  y1 - cover, -z1 + cover)
        ops.layer('straight', 2, 2, As_corner,
                  -y1 + cover, z1 - cover, -y1 + cover, -z1 + cover)
    else:
        # 3 bars top + 3 bars bottom
        ops.layer('straight', 2, 3, As_top / 3,
                   y1 - cover, z1 - cover,  y1 - cover, -z1 + cover)
        ops.layer('straight', 2, 3, As_bot / 3,
                  -y1 + cover, z1 - cover, -y1 + cover, -z1 + cover)

    # NO Aggregator wrapping — torsion is already in via -GJ.

# --- Columns ---
build_fiber_section(101, 350, 350, 40, 490, 490, is_col=True)   # Perimeter
build_fiber_section(102, 375, 375, 40, 562, 562, is_col=True)   # Interior

# --- Beams (one section per storey level) ---
build_fiber_section(201, 450, 300, 30, 600, 394)   # Plinth
build_fiber_section(202, 375, 230, 30, 600, 248)   # Floor 1
build_fiber_section(203, 325, 230, 30, 500, 211)   # Floor 2
build_fiber_section(204, 300, 230, 30, 400, 193)   # Floor 3
build_fiber_section(205, 250, 230, 30, 300, 157)   # Terrace / HR


def get_beam_sec(z_idx):
    if z_idx == 1: return 201
    if z_idx == 2: return 202
    if z_idx == 3: return 203
    if z_idx == 4: return 204
    return 205

# ============================================================================
# 4. NODES & BOUNDARY CONDITIONS
# ============================================================================
def get_node_tag(ix, iy, iz):
    return iz * 1000 + iy * 100 + ix + 1


def in_void(ix, iy):
    """Grid nodes inside the staircase void (not generated above base)."""
    return (3 <= ix <= 4) and (iy == 0)


def is_void_cell(cell_ix, cell_iy):
    """A slab cell between grid points (cx, cy) and (cx+1, cy+1)
    is part of the staircase void (no slab here)."""
    return (2 <= cell_ix <= 4) and (cell_iy == 0)


# --- Generate nodes ---
for iz, z in enumerate(Z_grid):
    for iy, y in enumerate(Y_grid):
        for ix, x in enumerate(X_grid):
            if in_void(ix, iy) and iz > 0:
                continue                          # void exists only above base
            node_tag = get_node_tag(ix, iy, iz)
            ops.node(node_tag, x, y, z)
            if iz == 0:
                ops.fix(node_tag, 1, 1, 1, 1, 1, 1)   # fully fixed base

# --- Rigid diaphragms (one master per floor) ---
master_nodes = {}
for iz in range(1, len(Z_grid)):
    master_tag = iz * 1000 + 999
    ops.node(master_tag,
             float(np.mean(X_grid)), float(np.mean(Y_grid)),
             float(Z_grid[iz]))
    # Master fixity:
    #   x, y free (free to translate horizontally)
    #   z FIXED (master doesn't move vertically — slaves still do via columns)
    #   Rx, Ry FIXED (out-of-plane rotations not used by diaphragm)
    #   Rz free (in-plane rotation = floor twist)
    ops.fix(master_tag, 0, 0, 1, 1, 1, 0)
    master_nodes[iz] = master_tag

    slave_tags = [get_node_tag(ix, iy, iz)
                  for iy in range(len(Y_grid))
                  for ix in range(len(X_grid))
                  if not in_void(ix, iy)]
    ops.rigidDiaphragm(3, master_tag, *slave_tags)

# ============================================================================
# 5. ELEMENTS  (FIX #2: X-beam transform corrected)
# ============================================================================
ops.geomTransf('PDelta', 1, 1, 0, 0)    # Columns      → local-y along −Y (square cols, OK)
ops.geomTransf('Linear', 2, 0, -1, 0)   # X-beams      → local-y = +Z (FIXED: was 0,1,0)
ops.geomTransf('Linear', 3, 1, 0, 0)    # Y-beams      → local-y = +Z (already correct)

ele_tag = 1

# --- Columns ---
for iz in range(len(Z_grid) - 1):
    for iy in range(len(Y_grid)):
        for ix in range(len(X_grid)):
            if in_void(ix, iy):
                continue
            nI = get_node_tag(ix, iy, iz)
            nJ = get_node_tag(ix, iy, iz + 1)
            perim = (ix == 0 or ix == len(X_grid) - 1 or
                     iy == 0 or iy == len(Y_grid) - 1)
            sec_tag = 101 if perim else 102
            ops.element('nonlinearBeamColumn', ele_tag, nI, nJ,
                        5, sec_tag, 1)
            ele_tag += 1

# --- Beams ---
for iz in range(1, len(Z_grid)):
    sec_tag = get_beam_sec(iz)

    # X-direction beams
    for iy in range(len(Y_grid)):
        for ix in range(len(X_grid) - 1):
            if (2 <= ix <= 4) and iy == 0:        # spans / borders void
                continue
            nI = get_node_tag(ix,     iy, iz)
            nJ = get_node_tag(ix + 1, iy, iz)
            ops.element('nonlinearBeamColumn', ele_tag, nI, nJ,
                        5, sec_tag, 2)
            ele_tag += 1

    # Y-direction beams
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
# STORY ELEVATIONS & GRAVITY LOAD TABLE
# ============================================================================
story_gravity_loads = {                 # total floor load in kN
    1.25 : 191,    # GROUND   – slab on grade; only structural self-wt
    4.50 : 1940,   # FLOOR 1  – typical floor
    7.75 : 1940,   # FLOOR 2  – typical floor
    11.00: 1940,   # FLOOR 3  – typical floor
    14.25: 1850,   # TERRACE  – reduced LL + waterproofing
    17.50: 650,    # HR SLAB  – partial slab
}

# ============================================================================
# 6. GRAVITY ANALYSIS  (FIX #1 — loads now go to SLAVE nodes by tributary)
# ============================================================================

def tributary_area(ix, iy):
    """Tributary slab area (mm²) for the node at grid (ix, iy).

    Each of the 4 adjacent slab cells contributes A/4 if that cell exists
    (i.e. is not part of the staircase void) and lies inside the floor plan.
    """
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
    elev = round(Z_grid[iz] / m, 2)
    W_total_N = story_gravity_loads.get(elev, 1940) * kN   # convert kN → N

    # Build slave-node list with each one's tributary area
    slave_info = []
    A_sum = 0.0
    for iy in range(len(Y_grid)):
        for ix in range(len(X_grid)):
            if in_void(ix, iy):
                continue
            A = tributary_area(ix, iy)
            if A > 0:
                slave_info.append((get_node_tag(ix, iy, iz), A))
                A_sum += A

    # Distribute W_total proportional to tributary area
    for node_tag, A_node in slave_info:
        W_node = W_total_N * (A_node / A_sum)
        ops.load(node_tag, 0.0, 0.0, -W_node, 0.0, 0.0, 0.0)

    total_applied += W_total_N
    print(f"  Elev {elev:>5.2f} m | {W_total_N/kN:>5.0f} kN  →  "
          f"{len(slave_info)} nodes (trib area = {A_sum/1e6:>6.2f} m²)")

# --- Solver settings for gravity ---
ops.system('BandGeneral')
ops.numberer('RCM')
ops.constraints('Transformation')
ops.test('NormDispIncr', 1.0e-4, 100)   # FIX #5: relaxed from 1e-6
ops.algorithm('Newton')
ops.integrator('LoadControl', 0.1)
ops.analysis('Static')
ok = ops.analyze(10)
if ok != 0:
    raise RuntimeError("Gravity analysis failed to converge")

ops.loadConst('-time', 0.0)             # freeze gravity, reset pseudo-time

print(f"\n✅ Gravity analysis complete. Total applied = "
      f"{total_applied/kN:.0f} kN ({total_applied/kN/1000:.2f} MN)")

# Sanity check: read base reactions; vertical component should ≈ gravity
ops.reactions()
base_nodes = [get_node_tag(ix, iy, 0)
              for ix in range(len(X_grid))
              for iy in range(len(Y_grid))
              if not in_void(ix, iy)]
Rz_sum = sum(ops.nodeReaction(n, 3) for n in base_nodes)
print(f"   Σ vertical base reactions = {Rz_sum/kN:.0f} kN "
      f"(should ≈ +{total_applied/kN:.0f} kN)")

# ============================================================================
# 7. CYCLIC PUSHOVER ANALYSIS
# ============================================================================
ops.wipeAnalysis()
ops.system('BandGeneral')
ops.numberer('RCM')
ops.constraints('Transformation')
ops.test('NormDispIncr', 1.0e-4, 200)
ops.algorithm('Newton')
ops.analysis('Static')

amps     = [2, 4, 8, 12, 16, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # mm
n_cycles = 3                                # cycles per amplitude

protocol = []
for amp in amps:
    for _ in range(n_cycles):
        protocol.extend([amp, -amp])

roof_master_node = master_nodes[len(Z_grid) - 1]

# Lateral load pattern: unit X-force at every floor master node.
# DisplacementControl scales this magnitude — values here are just shape.
ops.timeSeries('Linear', 2)
ops.pattern('Plain', 2, 2)
for iz, mtag in master_nodes.items():
    ops.load(mtag, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _try_algs():
    ok = ops.analyze(1)
    if ok == 0:
        return 0
    for alg in ('ModifiedNewton', 'KrylovNewton', 'NewtonLineSearch'):
        ops.algorithm(alg)
        ok = ops.analyze(1)
        if ok == 0:
            ops.algorithm('Newton')
            return 0
    ops.algorithm('Newton')
    return ok


def advance(dU_target, node_tag, dof=1, min_factor=64):
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


disp_history  = []
shear_history = []
peak_indices  = []
current_disp  = 0.0

print(f"\nStarting Cyclic Pushover (RCC, {len(protocol)} peaks)…")
for k, peak in enumerate(protocol):
    target_disp = peak * mm

    amp_now   = abs(peak)
    step_size = max(0.25, amp_now / 40.0) * mm

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

# .npz
np.savez(os.path.join(out_dir, 'disp_shear_rcc.npz'),
         disp=disp_array,
         shear=shear_array,
         peak_indices=np.asarray(peak_indices, dtype=int),
         amps=np.asarray(amps),
         n_cycles=n_cycles)

# Capacity envelope = (disp, shear) at every protocol peak
env_disp  = disp_array[peak_indices]
env_shear = shear_array[peak_indices]

# CSV — hysteresis (full time-history)
np.savetxt(os.path.join(out_dir, 'RCC_hysteresis.csv'),
           np.column_stack([disp_array, shear_array]),
           delimiter=',', header='Displacement_mm,BaseShear_kN', comments='')

# CSV — envelope (peak points only) + drift + secant stiffness
drift_env = env_disp / Z_grid[-1] * 100.0
stiff_env = np.where(np.abs(env_disp) > 0,
                     np.abs(env_shear) / np.abs(env_disp), 0.0)
np.savetxt(os.path.join(out_dir, 'RCC_envelope.csv'),
           np.column_stack([env_disp, env_shear, drift_env, stiff_env]),
           delimiter=',',
           header='Displacement_mm,BaseShear_kN,Drift_pct,Stiffness_kNmm',
           comments='')

# MATLAB plotting script
matlab_script = (
    "% RCC_plots.m  -- auto-generated\n"
    "hyst = readmatrix('RCC_hysteresis.csv');\n"
    "env  = readmatrix('RCC_envelope.csv');\n"
    "d = hyst(:,1); V = hyst(:,2);\n"
    "ed = env(:,1); eV = env(:,2); drift = env(:,3); K = env(:,4);\n\n"
    "figure; plot(d, V, 'r', 'LineWidth', 1); grid on;\n"
    "xlabel('Lateral Displacement (mm)'); ylabel('Lateral Load (kN)');\n"
    "title('Hysteretic Response (RCC)');\n\n"
    "figure; plot(ed, eV, 'b-o', 'MarkerSize', 4); grid on;\n"
    "xlabel('Lateral Displacement (mm)'); ylabel('Lateral Load (kN)');\n"
    "title('Capacity Envelope (RCC)');\n\n"
    "[~,idx] = sort(abs(ed));\n"
    "figure; plot(abs(ed(idx)), K(idx), 'g-s', 'MarkerSize', 4); grid on;\n"
    "xlabel('Lateral Displacement (mm)'); ylabel('Stiffness (kN/mm)');\n"
    "title('Stiffness Degradation (RCC)');\n\n"
    "pos = ed > 0;\n"
    "figure; plot(drift(pos), eV(pos), 'm-o', 'MarkerSize', 4); grid on;\n"
    "xlabel('Storey Drift (%)'); ylabel('Lateral Load (kN)');\n"
    "title('Capacity Curve (RCC)');\n"
)
with open(os.path.join(out_dir, 'RCC_plots.m'), 'w', encoding='utf-8') as fh:
    fh.write(matlab_script)

# Stiffness degradation arrays (sorted by |disp| for clean plot)
stiff_disp = np.abs(env_disp)
stiff_vals = np.where(stiff_disp > 0,
                      np.abs(env_shear) / np.where(stiff_disp == 0, 1, stiff_disp),
                      0.0)
order = np.argsort(stiff_disp)
stiff_disp, stiff_vals = stiff_disp[order], stiff_vals[order]

plt.style.use('ggplot')

# Plot 1: Hysteresis
plt.figure(figsize=(10, 6))
plt.plot(disp_array, shear_array, color='red', linewidth=1.0,
         label='Finite Element Frame')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Hysteretic Load–Displacement Response Curve (RCC)')
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Lateral Load (kN)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Hysteresis_RCC.png'), dpi=150)

# Plot 2: Capacity Envelope
plt.figure(figsize=(10, 6))
plt.plot(env_disp, env_shear, marker='o', color='blue',
         markersize=4, linestyle='-', label='Capacity Envelope')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Load–Displacement Envelope Curve (RCC)')
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Lateral Load (kN)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Capacity_Envelope_RCC.png'), dpi=150)

# Plot 3: Stiffness Degradation
plt.figure(figsize=(10, 6))
plt.plot(stiff_disp, stiff_vals, marker='s', color='green',
         markersize=4, linestyle='-', label='Stiffness Degradation')
plt.title('Stiffness Degradation Curve (RCC)')
plt.xlabel('Lateral Displacement (mm)')
plt.ylabel('Stiffness (kN/mm)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Stiffness_Degradation_RCC.png'), dpi=150)

# Plot 4: Capacity Curve (drift % vs shear) — positive pushover direction only
pos_mask  = env_disp > 0
cap_drift = env_disp[pos_mask] / Z_grid[-1] * 100.0
cap_shear = env_shear[pos_mask]
plt.figure(figsize=(10, 6))
plt.plot(cap_drift, cap_shear, marker='o', color='purple',
         markersize=4, linestyle='-', label='Capacity Curve')
plt.title('Capacity Curve – Load vs Storey Drift (RCC)')
plt.xlabel('Storey Drift (%)')
plt.ylabel('Lateral Load (kN)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Capacity_Curve_RCC.png'), dpi=150)

print(f"\n✅ RCC analysis complete. Results saved to {out_dir}")