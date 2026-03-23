import numpy as np
import matplotlib.pyplot as plt
from standard_regime import prandtl_tip_root, solve_section

R = 0.7
B = 6
h = 2000
R0 = 0.25 * R
U0 = 60.0

rho = 1.225 * (1 - 2.25577e-5 * h) ** 4.256
D_prop = 2 * R
collective_pitch = 46

data = np.loadtxt("ARAD8pct_polar.txt", comments="#")
alpha_tab = data[:, 0]
cl_tab = data[:, 1]
cd_tab = data[:, 2]

def Cl(alpha_deg):
    return np.interp(alpha_deg, alpha_tab, cl_tab)

def Cd(alpha_deg):
    return np.interp(alpha_deg, alpha_tab, cd_tab)

def beta(mu, collective_pitch=46):
    return np.radians(-50.0 * mu + 35 + collective_pitch)

def chord(mu):
    return R * (0.18 - 0.06 * mu)

def evaluate_propeller(J, collective_pitch=46, n_annuli=60):
    n_rev = U0 / (J * D_prop)
    RPM = 60.0 * n_rev
    omega = 2.0 * np.pi * n_rev
    lam = omega * R / U0

    mu_root = R0 / R
    mu_arr_sides = np.linspace(mu_root, 1, n_annuli)
    mu_array = np.zeros(n_annuli-1)

    dT_arr = np.zeros(n_annuli-1)
    dQ_arr = np.zeros(n_annuli-1)
    any_aoa_out_of_bounds = False

    for i in range(len(mu_arr_sides)-1):
        sol = solve_section(mu_arr_sides[i], mu_arr_sides[i+1], omega=omega, collective_pitch=collective_pitch)
        mu = (mu_arr_sides[i]+ mu_arr_sides[i+1]) / 2.0
        mu_array[i] = mu
        r = mu * R
        dT_arr[i] = B * sol["Fax_blade"] * R*(mu_arr_sides[i+1] - mu_arr_sides[i])
        dQ_arr[i] = B * sol["Ftan_blade"] * r * R*(mu_arr_sides[i+1] - mu_arr_sides[i])
        if not sol["aoa_in_bounds"]:
            any_aoa_out_of_bounds = True

    Thrust = np.sum(dT_arr)
    Torque = np.sum(dQ_arr)
    Power = omega * Torque

    CT = Thrust / (rho * n_rev**2 * D_prop**4)
    CP = Power / (rho * n_rev**3 * D_prop**5)
    Tc = Thrust / (rho * U0**2 * (2*R)**2)
    Pc = Power / (rho * U0**3 * (2*R)**2)
    eta_prop = (J * CT / CP) if CP > 0 else np.nan
    eta_harv = -Pc * 8/np.pi if Pc < 0 else np.nan

    return {
        "J": J,
        "RPM": RPM,
        "omega": omega,
        "lambda": lam,
        "Thrust": Thrust,
        "Torque": Torque,
        "Power": Power,
        "CT": CT,
        "CP": CP,
        "Tc": Tc,
        "Pc": Pc,
        "eta_prop": eta_prop,
        "eta_harv": eta_harv,
        "any_aoa_out_of_bounds": any_aoa_out_of_bounds
    }

# Validation propulsive regime
J_vals = np.linspace(0.3, 2.6, 40)

CT_vals = []
CP_vals = []
Tc_vals = []
Pc_vals = []
eta_prop_vals = []
eta_harv_vals = []
eta_Jin=[]
aoa_out_vals = []

for J in J_vals:
    res = evaluate_propeller(J)
    CT_vals.append(res["CT"])
    CP_vals.append(res["CP"])
    Tc_vals.append(res["Tc"])
    Pc_vals.append(res["Pc"])
    eta_prop_vals.append(res["eta_prop"])
    eta_harv_vals.append(res["eta_harv"])
    aoa_out_vals.append(res["any_aoa_out_of_bounds"])

    print(f"J={J:.3f}, CT={res['CT']:.5f}, CP={res['CP']:.5f}, eta_prop={res['eta_prop']:.5f}, eta_harv={res['eta_harv']:.5f}")

CT_vals = np.array(CT_vals)
CP_vals = np.array(CP_vals)
Tc_vals = np.array(Tc_vals)
Pc_vals = np.array(Pc_vals)
eta_prop_vals = np.array(eta_prop_vals)
eta_harv_vals = np.array(eta_harv_vals)
aoa_out_vals = np.array(aoa_out_vals)


def load_javafoil_data(filename):
    """
    Load JavaFoil output data from a text file.

    Parameters:
        filename : str
            Path to the JavaFoil .txt file.

    Returns:
        dict :
            Dictionary containing arrays:
            - 'J'  : advance ratio (v/(nD))
            - 'CT' : thrust coefficient
            - 'CP' : power coefficient
            - 'eta': propeller efficiency
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Skip empty lines or comment lines
            if not line.strip() or line.startswith('[') or line.startswith('v/'):
                continue
            # Split line into columns
            cols = line.split()
            if len(cols) < 9:  # make sure enough columns
                continue
            # Extract relevant columns
            try:
                J = float(cols[0])
                CT = float(cols[2])
                CP = float(cols[3])
                TC = float(cols[5])*np.pi/8  # Thrust coefficient (if needed)
                PC = float(cols[6])*np.pi/8  # Power coefficient (if needed)
                eta = float(cols[7]) / 100.0  # convert % to 0-1
                data.append([J, CT, CP, eta,TC,PC])
            except ValueError:
                continue

    data = np.array(data)
    return {
        'J': data[:,0],
        'CT': data[:,1],
        'CP': data[:,2],
        'eta': data[:,3],
        'TC': data[:,4],
        'PC': data[:,5]
    }


# ===============================
# 2️⃣ Example: Plot my data vs JavaFoil
# ===============================

# Load JavaFoil results
jf_data = load_javafoil_data("javaprop_validation_plots.txt")

# Assume your computed data arrays:
# J_vals, CT_vals, CP_vals, eta_vals

plt.figure()
plt.plot(jf_data['J'], jf_data['CT'], 's--', color='black', label='JavaProp', zorder=1)
plt.plot(J_vals, CT_vals, 'o-', label='BEM implementation', zorder=2)
if aoa_out_vals.any():
    plt.plot(J_vals[aoa_out_vals], CT_vals[aoa_out_vals], 'ro', label='AOA out of bounds', zorder=3)
plt.xlabel("Advance ratio J")
plt.ylabel(r"$C_T$")
plt.title("Thrust Coefficient vs Advance Ratio")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(jf_data['J'], jf_data['CP'], 's--', color='black', label='JavaProp', zorder=1)
plt.plot(J_vals, CP_vals, 'o-', label='BEM implementation', zorder=2)
if aoa_out_vals.any():
    plt.plot(J_vals[aoa_out_vals], CP_vals[aoa_out_vals], 'ro', label='AOA out of bounds', zorder=3)
plt.xlabel("Advance ratio J")
plt.ylabel(r"$C_P$")
plt.title("Power Coefficient vs Advance Ratio")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(jf_data['J'], jf_data['eta'], 's--', color='black', label='JavaProp', zorder=1)
plt.plot(J_vals, eta_prop_vals, 'o-', label='BEM implementation', zorder=2)
if aoa_out_vals.any():
    plt.plot(J_vals[aoa_out_vals], eta_prop_vals[aoa_out_vals], 'ro', label='AOA out of bounds', zorder=3)
plt.xlabel("Advance ratio J")
plt.ylabel(r"$\eta_{prop}$")
plt.title("Propeller Efficiency vs Advance Ratio")
plt.legend()
plt.grid(True)

plt.show()

plt.figure()
plt.plot(jf_data['J'], jf_data['TC'], 's--', color='black', label='JavaProp', zorder=1)
plt.plot(J_vals, Tc_vals, 'o-', label='BEM implementation', zorder=2)
if aoa_out_vals.any():
    plt.plot(J_vals[aoa_out_vals], Tc_vals[aoa_out_vals], 'ro', label='AOA out of bounds', zorder=3)
plt.xlabel("Advance ratio J")
plt.ylabel(r"$T_c$")
plt.title("Tc vs Advance Ratio")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(jf_data['J'], jf_data['PC'], 's--', color='black', label='JavaProp', zorder=1)
plt.plot(J_vals, Pc_vals, 'o-', label='BEM implementation', zorder=2)
if aoa_out_vals.any():
    plt.plot(J_vals[aoa_out_vals], Pc_vals[aoa_out_vals], 'ro', label='AOA out of bounds', zorder=3)
plt.xlabel("Advance ratio J")
plt.ylabel(r"$P_c$")
plt.title("Pc vs Advance Ratio")
plt.legend()
plt.grid(True)
plt.show()

# Validation energy harvesting regime
# J_vals = np.linspace(2, 3, 40)

# CT_vals = []
# CP_vals = []
# Tc_vals = []
# Pc_vals = []
# eta_prop_vals = []
# eta_harv_vals = []
# eta_Jin = []

# for J in J_vals:
#     res = evaluate_propeller(J)
#     CT_vals.append(res["CT"])
#     CP_vals.append(res["CP"])
#     Tc_vals.append(res["Tc"])
#     Pc_vals.append(res["Pc"])
#     eta_prop_vals.append(res["eta_prop"])
#     eta_harv_vals.append(res["eta_harv"])
#     eta_Jin.append(res["CP"] / (J * res["CT"]))
#     print(f"J={J:.3f}, CT={res['CT']:.5f}, CP={res['CP']:.5f}, eta_prop={res['eta_prop']:.5f}, eta_harv={res['eta_harv']:.5f}, eta_Jin={eta_Jin[-1]:.5f}")

# CT_vals = np.array(CT_vals)
# CP_vals = np.array(CP_vals)
# Tc_vals = np.array(Tc_vals)
# Pc_vals = np.array(Pc_vals)
# eta_prop_vals = np.array(eta_prop_vals)
# eta_harv_vals = np.array(eta_harv_vals)
# eta_Jin = np.array(eta_Jin)


# def load_javafoil_data(filename):
#     """
#     Load JavaFoil output data from a text file.

#     Parameters:
#         filename : str
#             Path to the JavaFoil .txt file.

#     Returns:
#         dict :
#             Dictionary containing arrays:
#             - 'J'  : advance ratio (v/(nD))
#             - 'CT' : thrust coefficient
#             - 'CP' : power coefficient
#             - 'eta': propeller efficiency
#     """
#     data = []
#     with open(filename, 'r') as f:
#         for line in f:
#             # Skip empty lines or comment lines
#             if not line.strip() or line.startswith('[') or line.startswith('v/'):
#                 continue
#             # Split line into columns
#             cols = line.split()
#             if len(cols) < 9:  # make sure enough columns
#                 continue
#             # Extract relevant columns
#             try:
#                 J = float(cols[0])
#                 CT = float(cols[2])
#                 CP = float(cols[3])
#                 TC = float(cols[5])*np.pi/8  # Thrust coefficient (if needed)
#                 PC = float(cols[6])*np.pi/8  # Power coefficient (if needed)
#                 eta = float(cols[7]) / 100.0  # convert % to 0-1
#                 data.append([J, CT, CP, eta,TC,PC])
#             except ValueError:
#                 continue

#     data = np.array(data)
#     return {
#         'J': data[:,0],
#         'CT': data[:,1],
#         'CP': data[:,2],
#         'eta': data[:,3],
#         'TC': data[:,4],
#         'PC': data[:,5]
#     }


# # ===============================
# # 2️⃣ Example: Plot my data vs JavaFoil
# # ===============================

# # Load JavaFoil results
# jf_data = load_javafoil_data("javaprop_validation_2.txt")

# # Assume your computed data arrays:
# # J_vals, CT_vals, CP_vals, eta_vals

# plt.figure()
# plt.plot(J_vals, CT_vals, 'o-', label='BEM implementation')
# plt.plot(jf_data['J'], jf_data['CT'], 's--', label='JavaFoil')
# plt.xlabel("Advance ratio J")
# plt.ylabel(r"$C_T$")
# plt.title("Thrust Coefficient vs Advance Ratio")
# plt.legend()
# plt.grid(True)

# plt.figure()
# plt.plot(J_vals, CP_vals, 'o-', label='My BEM')
# plt.plot(jf_data['J'], jf_data['CP'], 's--', label='JavaFoil')
# plt.xlabel("Advance ratio J")
# plt.ylabel(r"$C_P$")
# plt.title("Power Coefficient vs Advance Ratio")
# plt.legend()
# plt.grid(True)

# plt.figure()
# plt.plot(J_vals, eta_harv_vals, 'o-', label='My BEM')
# plt.plot(jf_data['J'], -8*jf_data['PC']/np.pi, 's--', label='JavaFoil')
# plt.xlabel("Advance ratio J")
# plt.ylabel(r"$\eta_{harv}$")
# plt.title("Harvesting Efficiency vs Advance Ratio")
# plt.legend()
# plt.grid(True)

# plt.show()

# plt.figure()
# plt.plot(J_vals, Tc_vals, 'o-', label='My BEM')
# plt.plot(jf_data['J'], jf_data['TC'], 's--', label='JavaFoil')
# plt.xlabel("Advance ratio J")
# plt.ylabel(r"$C_T$")
# plt.title("Thrust Coefficient vs Advance Ratio")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure()
# plt.plot(J_vals, Pc_vals, 'o-', label='My BEM')
# plt.plot(jf_data['J'], jf_data['PC'], 's--', label='JavaFoil')
# plt.xlabel("Advance ratio J")
# plt.ylabel(r"$P_C$")
# plt.title("Power Coefficient vs Advance Ratio")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure()
# plt.plot(J_vals, eta_Jin, 'o-', label='My BEM')
# plt.xlabel("Advance ratio J")
# plt.ylabel(r"$\eta_{Jin}$")
# plt.title("Inflow Efficiency vs Advance Ratio")
# plt.legend()
# plt.grid(True)
# plt.show()

# # ======================================
# # Combined CT & CP plot with lambda axis
# # ======================================

J_plot = np.linspace(1, 3.5, 20)

CT_plot = []
CP_plot = []
lambda_plot = []
aoa_out_plot = []

for J in J_plot:
    res = evaluate_propeller(J,46)
    CT_plot.append(res["CT"])
    CP_plot.append(res["CP"])
    lambda_plot.append(res["lambda"])
    aoa_out_plot.append(res["any_aoa_out_of_bounds"])

CT_plot = np.array(CT_plot)
CP_plot = np.array(CP_plot)
lambda_plot = np.array(lambda_plot)
aoa_out_plot = np.array(aoa_out_plot)

# --- Create figure ---
fig, ax1 = plt.subplots()

# Plot CT and CP
ax1.plot(J_plot, CT_plot, '-', color='black', label=r"$C_T$")
ax1.plot(J_plot, CP_plot, '-', color='gray', label=r"$C_P$")
if aoa_out_plot.any():
    ax1.plot(J_plot[aoa_out_plot], CT_plot[aoa_out_plot], 'ro')
    ax1.plot(J_plot[aoa_out_plot], CP_plot[aoa_out_plot], 'ro', label="AOA out of polar bounds")

ax1.set_xlabel("Advance ratio $J$")
ax1.set_ylabel(r"Thrust, Power coefficient $C_T, C_P$")
ax1.grid(True)

# Zero lines
ax1.axhline(0, color='black', linewidth=1)

div_color = 'steelblue'
div_style = dict(color=div_color, linewidth=1.2, linestyle='--')

J_min, J_max = J_plot[0], J_plot[-1]

# # Vertical line at CT = 0 crossing
# CT_crossings = np.where(np.diff(np.sign(CT_plot)))[0]
# J_CT0 = None
# if len(CT_crossings) > 0:
#     i = CT_crossings[0]
#     J_CT0 = J_plot[i] - CT_plot[i] * (J_plot[i+1] - J_plot[i]) / (CT_plot[i+1] - CT_plot[i])
#     ax1.axvline(J_CT0, **div_style)

# # Vertical line at CP = 0 crossing
# CP_crossings = np.where(np.diff(np.sign(CP_plot)))[0]
# J_CP0 = None
# if len(CP_crossings) > 0:
#     i = CP_crossings[0]
#     J_CP0 = J_plot[i] - CP_plot[i] * (J_plot[i+1] - J_plot[i]) / (CP_plot[i+1] - CP_plot[i])
#     ax1.axvline(J_CP0, **div_style)

# # --- Regime labels at the top of each region (axes fraction y) ---
# # Three regions: propulsive (J < J_CT0), braking (J_CT0 < J < J_CP0), harvesting (J > J_CP0)
# label_y = 0.05  # near bottom, in axes fraction
# label_kw = dict(transform=ax1.transAxes, ha='center', va='bottom',
#                 fontsize=9, color=div_color,
#                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

# def _x_to_axes(x):
#     """Convert data x-coordinate to axes fraction."""
#     return (x - J_min) / (J_max - J_min)

# # Propulsive regime: left edge to J_CT0 (or J_CP0 if J_CT0 missing)
# left_bound  = J_CT0 if J_CT0 is not None else J_CP0
# right_bound = J_CP0 if J_CP0 is not None else J_CT0

# if left_bound is not None:
#     x_prop = _x_to_axes((J_min + left_bound) / 2)
#     ax1.text(x_prop, label_y, "propulsive\nregime", **label_kw)

# if J_CT0 is not None and J_CP0 is not None:
#     x_brake = _x_to_axes((J_CT0 + J_CP0) / 2)
#     ax1.text(x_brake, label_y, "braking\nregime", **label_kw)

# if right_bound is not None:
#     x_harv = _x_to_axes((right_bound + J_max) / 2)
#     ax1.text(x_harv, label_y, "energy\nharvesting", **label_kw)

# --- Top axis for lambda ---
ax2 = ax1.twiny()

# Match limits
ax2.set_xlim(ax1.get_xlim())

# Select a few tick positions
J_ticks = np.linspace(1, 3.5, 6)
lambda_ticks = []

for J in J_ticks:
    res = evaluate_propeller(J)
    lambda_ticks.append(res["lambda"])

ax2.set_xticks(J_ticks)
ax2.set_xticklabels([f"{lam:.2f}" for lam in lambda_ticks])
ax2.set_xlabel(r"Tip speed ratio $\lambda$")

# Legend
ax1.legend()

fig.tight_layout()
fig.subplots_adjust(top=0.88)  # extra headroom for lambda axis labels
# plt.title("Propeller Performance Map")
plt.show()

#Pitch optimization
def optimize_collective_pitch(pitch_range, J_range):
    global collective_pitch

    best_pitch = None
    best_min_Pc = np.inf
    best_results = None

    pitch_summary = []  # one entry per pitch: {pitch, min_Pc, any_out, results}

    for pitch in pitch_range:
        collective_pitch = pitch

        results_store = []

        for J in J_range:
            res = evaluate_propeller(J, collective_pitch=pitch)
            results_store.append(res)

        # Only use J points where all blade sections are within polar bounds
        valid_Pc = [res["Pc"] for res in results_store if not res["any_aoa_out_of_bounds"]]
        any_out = any(res["any_aoa_out_of_bounds"] for res in results_store)

        if valid_Pc:
            min_Pc = np.min(valid_Pc)
        else:
            min_Pc = np.nan  # all J points were out of bounds for this pitch

        print(f"Pitch: {pitch:.2f} deg, Min Pc (valid only): {min_Pc:.5f}, any AOA out: {any_out}")

        pitch_summary.append({
            "pitch": pitch,
            "min_Pc": min_Pc,
            "any_out": any_out,
            "results": results_store,
        })

        # We want MOST NEGATIVE Pc → minimum value; skip if no valid points
        if not np.isnan(min_Pc) and min_Pc < best_min_Pc:
            best_min_Pc = min_Pc
            best_pitch = pitch
            best_results = results_store

    return best_pitch, best_min_Pc, best_results, pitch_summary

# Define ranges
pitch_range = np.linspace(20, 70, 5)   # adjust if needed
J_range = np.linspace(0.5, 4.0, 30)

best_pitch, best_min_Pc, best_results, pitch_summary = optimize_collective_pitch(pitch_range, J_range)

print(f"\nOptimal collective pitch: {best_pitch:.2f} deg")
print(f"Minimum Pc achieved: {best_min_Pc:.5f}")
J_opt = np.array([res["J"] for res in best_results])
Pc_opt = np.array([res["Pc"] for res in best_results])
Tc_opt = np.array([res["Tc"] for res in best_results])
eta_harv_opt = np.array([res["eta_harv"] for res in best_results])
aoa_out_opt = np.array([res["any_aoa_out_of_bounds"] for res in best_results])

out_b = aoa_out_opt

plt.figure()
plt.plot(J_opt, eta_harv_opt, 'o-', label=f"Pitch = {best_pitch:.1f}°")
if out_b.any():
    plt.plot(J_opt[out_b], eta_harv_opt[out_b], 'ro', label="AOA out of polar bounds")
plt.xlabel("Advance ratio J")
plt.ylabel(r"$\eta_{harv}$")
plt.title(f"Harvesting Efficiency (Pitch = {best_pitch:.1f}°)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Plot min Pc vs collective pitch (valid pitches only, no red dots since all values are from valid J)
# -------------------------
pitches = np.array([s["pitch"] for s in pitch_summary])
min_Pc_arr = np.array([s["min_Pc"] for s in pitch_summary])

valid_pitch = ~np.isnan(min_Pc_arr)

plt.figure()
plt.plot(pitches[valid_pitch], min_Pc_arr[valid_pitch], 'o-')
plt.xlabel("Collective pitch [deg]")
plt.ylabel(r"Min $P_c$")
plt.title("Minimum Power Coefficient vs Collective Pitch")
plt.grid(True)
plt.show()

# -------------------------
# Plot Pc vs J for all pitches in one figure
# -------------------------
fig, ax = plt.subplots()
colors = plt.cm.viridis(np.linspace(0, 1, len(pitch_summary)))
out_of_bounds_label_added = False

for s, color in zip(pitch_summary, colors):
    J_arr = np.array([res["J"] for res in s["results"]])
    Pc_arr = np.array([res["Pc"] for res in s["results"]])
    aoa_out = np.array([res["any_aoa_out_of_bounds"] for res in s["results"]])

    ax.plot(J_arr, Pc_arr, '-', color=color, label=f"Pitch = {s['pitch']:.1f}°")

    if aoa_out.any():
        label = "AOA out of polar bounds" if not out_of_bounds_label_added else "_nolegend_"
        ax.plot(J_arr[aoa_out], Pc_arr[aoa_out], 'ro', label=label)
        out_of_bounds_label_added = True

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel("Advance ratio J")
ax.set_ylabel(r"$P_c$")
ax.set_title("Power Coefficient vs J for All Collective Pitches")
ax.legend(loc="best", fontsize="small")
ax.grid(True)
plt.show()