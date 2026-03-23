import numpy as np
import matplotlib.pyplot as plt

# Please note that the induction factor has been defined in the opposite way as in the lectures, so that both a and ap are positive for the propeller and negative for the wind turbine.

# Defining constants 

R = 0.7 #Propeller radius in meters
B = 6 #Number of blades
h = 2000 #Altitude in meters
R0 = 0.25 * R #Blade root location in meters (before that is the hub, which we ignore)
U0 = 60.0 #Freestream velocity in m/s
J = 2.0 #Advance ratio 

rho = 1.225 * (1 - 2.25577e-5 * h) ** 4.256 #Denisty with altitude model

D_prop = 2 * R #Prop diameter in meters
n_rev = U0 / (J * D_prop) #RPS
RPM = 60.0 * n_rev #Revolutions per minute
omega = 2.0 * np.pi * n_rev #Prop angular velocity in rad/s
lam = omega * R / U0 #Tip speed ratio 

# Load airfoil data

data = np.loadtxt("ARAD8pct_polar.txt", comments="#")
alpha_tab = data[:, 0]
cl_tab = data[:, 1]
cd_tab = data[:, 2]

# Interpolation functions for Cl and Cd based on the airfoil data

def Cl(alpha_deg):
    if alpha_deg < alpha_tab[0] or alpha_deg > alpha_tab[-1]:
        alpha_deg = np.clip(alpha_deg, alpha_tab[0], alpha_tab[-1])
    return np.interp(alpha_deg, alpha_tab, cl_tab)

def Cd(alpha_deg):
    if alpha_deg < alpha_tab[0] or alpha_deg > alpha_tab[-1]:
        alpha_deg = np.clip(alpha_deg, alpha_tab[0], alpha_tab[-1])
    return np.interp(alpha_deg, alpha_tab, cd_tab)


# Defining functions describing blade geometry

def beta(mu, collective_pitch=46): # mu = r/R
    return np.radians(-50.0 * mu + 35 + collective_pitch)

def chord(mu): # mu = r/R
    return R * (0.18 - 0.06 * mu)

# Prandtl's correction for finite number of blades, including tip and root losses

def prandtl_tip_root(mu, a, ap, lam_local=None): # mu = r/R (radial position), a = axial induction factor, ap = tangential induction factor
    mu_root = R0 / R
    if lam_local is None:
        lam_local = lam
    # mu = np.clip(mu, mu_root + 1e-8, 1.0 - 1e-8)

    # denom = max(1.0 + a, 1e-8)
    denom = 1+a
    sqrt_term = np.sqrt(1.0 + ((lam_local * mu) / denom) ** 2)

    expo_tip = -(B / 2.0) * ((1.0 - mu) / mu) * sqrt_term
    expo_root = -(B / 2.0) * ((mu - mu_root) / mu) * sqrt_term

    # f_tip_arg = np.exp(np.clip(expo_tip, -700.0, 0.0))
    # f_root_arg = np.exp(np.clip(expo_root, -700.0, 0.0))
    f_tip_arg = np.exp(expo_tip)
    f_root_arg = np.exp(expo_root)

    # f_tip_arg = np.clip(f_tip_arg, 0.0, 1.0)
    # f_root_arg = np.clip(f_root_arg, 0.0, 1.0)

    f_tip = (2.0 / np.pi) * np.arccos(f_tip_arg)
    f_root = (2.0 / np.pi) * np.arccos(f_root_arg)

    F = f_tip * f_root
    return np.clip(F, 1e-4, 1.0), f_tip, f_root
    # return F, f_tip, f_root

def CT_from_a(a, glauert=False):
    CT = 4 * a * (1 - a)

    if glauert:
        CT1 = 1.816
        a1 = 1 - np.sqrt(CT1) / 2

        if a > a1:
            CT = CT1 - 4 * (np.sqrt(CT1) - 1) * (1 - a)

    return CT


def a_from_CT(CT):
    CT1 = 1.816
    CT2 = 2 * np.sqrt(CT1) - CT1

    if CT >= CT2:
        a = 1 + (CT - CT1) / (4 * (np.sqrt(CT1) - 1))
    else:
        a = 0.5 - 0.5 * np.sqrt(max(1 - CT, 1e-8))

    return a


def solve_section(mu1, mu2, omega, collective_pitch=46,
                  a0=0.3, ap0=0.01, max_iter=1000, tol=1e-6, relax=0.05):

    mu = 0.5 * (mu1 + mu2)
    r = mu * R
    dr = R * (mu2 - mu1)

    Area = np.pi * R**2 * (mu2**2 - mu1**2)

    lam_local = omega * R / U0   # tip speed ratio at this operating point

    c_local = chord(mu)
    sigma = B * c_local / (2.0 * np.pi * r)

    a = a0
    ap = ap0
    converged = False

    for _ in range(max_iter):

        # --- velocities ---
        Vax = U0 * (1.0 + a)
        Vtan = omega * r * (1.0 - ap)

        phi = np.arctan2(Vax, Vtan)

        s = np.sin(phi)
        c = np.cos(phi)

        # --- airfoil ---
        alpha = beta(mu, collective_pitch) - phi
        alpha_deg = np.degrees(alpha)

        cl = Cl(alpha_deg)
        cd = Cd(alpha_deg)

        Cn = cl * c - cd * s
        Ct = cl * s + cd * c

        # --- Prandtl ---
        F, _, _ = prandtl_tip_root(mu, a, ap, lam_local)

        # --- relative velocity ---
        W = np.hypot(Vax, Vtan)
        q = 0.5 * rho * W**2

        # --- forces per unit span (per blade) ---
        L = q * cl * c_local
        D = q * cd * c_local

        f_ax = L * np.cos(phi) - D * np.sin(phi)
        f_tan = L * np.sin(phi) + D * np.cos(phi)

        # =========================
        # ✅ NEW: a' FROM FORCES
        # =========================

        # torque from annulus
        dQ = B * f_tan * dr * r

# --- tangential induction from forces (CORRECT BEM form) ---

        denom = 4.0 * np.pi * rho * U0 * (1.0 + a) * omega * r**2

        if abs(denom) < 1e-8 or F < 1e-6:
            ap_new = 0.0
        else:
            ap_new = B * f_tan / denom
            ap_new /= F

        # =========================
        # axial induction (unchanged)
        # =========================

        s2 = max(s * s, 1e-8)
        CT_loc = sigma * Cn / (F * s2)

        a_new = a_from_CT(CT_loc)

        # --- relaxation ---
        if abs(a_new - a) < tol and abs(ap_new - ap) < tol:
            a = a_new
            ap = ap_new
            converged = True
            break

        a = (1 - relax) * a + relax * a_new
        ap = (1 - relax) * ap + relax * ap_new

    if not converged:
        print(f"Warning: section at mu={mu:.4f} did not converge")

    # --- final forces ---
    Vax = U0 * (1.0 + a)
    Vtan = omega * r * (1.0 - ap)
    phi = np.arctan2(Vax, Vtan)

    alpha = beta(mu, collective_pitch) - phi
    alpha_deg = np.degrees(alpha)
    # if alpha_deg < alpha_tab[0] or alpha_deg > alpha_tab[-1]:
        # print(f"Warning: alpha {alpha_deg:.2f} out of range at mu={mu:.4f}")

    cl = Cl(alpha_deg)
    cd = Cd(alpha_deg)

    W = np.hypot(Vax, Vtan)
    q = 0.5 * rho * W**2

    L = q * cl * c_local
    D = q * cd * c_local

    dF_ax_per_blade = L * np.cos(phi) - D * np.sin(phi)
    dF_tan_per_blade = L * np.sin(phi) + D * np.cos(phi)

    F, ftip, froot = prandtl_tip_root(mu, a, ap, lam_local)

    aoa_in_bounds = (alpha_tab[0] <= alpha_deg <= alpha_tab[-1])

    return {
        "a": a,
        "ap": ap,
        "phi": phi,
        "alpha": alpha,
        "W": W,
        "Fax_blade": dF_ax_per_blade,
        "Ftan_blade": dF_tan_per_blade,
        "F": F,
        "ftip": ftip,
        "froot": froot,
        "aoa_in_bounds": aoa_in_bounds,
    }


def prop_performance(B, R, R0, U0, omega,collective_pitch, lam, n_annuli, solve_section):
    """
    Compute propeller performance using BEM (propeller-style convention).

    Parameters:
        B           : int       - number of blades
        R           : float     - propeller radius [m]
        R0          : float     - hub radius [m]
        U0          : float     - inflow velocity [m/s]
        collective_pitch : float - collective pitch angle [deg]
        omega       : float     - rotation rate [rad/s]
        n_annuli    : int       - number of blade elements

    Returns:
        dict with arrays: mu_arr, a_arr, ap_arr, phi_arr, alpha_arr, W_arr, F_arr,
                          dT_arr, dQ_arr, r_arr, Thrust, Torque, Power, eta
    """
    mu_arr_sides = np.linspace(R0 / R, 1.0, n_annuli)
    mu_arr = np.zeros(n_annuli - 1)
    r_arr = np.zeros(n_annuli - 1)
    a_arr = np.zeros(n_annuli-1)
    ap_arr = np.zeros(n_annuli-1)
    phi_arr = np.zeros(n_annuli-1)
    alpha_arr = np.zeros(n_annuli-1)
    W_arr = np.zeros(n_annuli-1)
    F_arr = np.zeros(n_annuli-1)
    aoa_in_bounds_arr = np.ones(n_annuli-1, dtype=bool)

    dT_arr = np.zeros(n_annuli-1)
    dQ_arr = np.zeros(n_annuli-1)

    for i in range(n_annuli - 1):
        sol = solve_section(mu_arr_sides[i], mu_arr_sides[i+1], omega,collective_pitch, lam)  # BEM solution for annulus
        mu = (mu_arr_sides[i] + mu_arr_sides[i+1]) / 2.0  # midpoint for storing results
        r = mu * R
        mu_arr[i] = mu
        r_arr[i] = r
        a_arr[i] = sol["a"]
        ap_arr[i] = sol["ap"]
        phi_arr[i] = np.degrees(sol["phi"])
        alpha_arr[i] = np.degrees(sol["alpha"])
        W_arr[i] = sol["W"]
        F_arr[i] = sol["F"]
        aoa_in_bounds_arr[i] = sol["aoa_in_bounds"]

        dT_arr[i] = B * sol["Fax_blade"] * R*(mu_arr_sides[i+1] - mu_arr_sides[i])
        dQ_arr[i] = B * sol["Ftan_blade"] * r * R*(mu_arr_sides[i+1] - mu_arr_sides[i])

    Thrust = np.sum(dT_arr)
    Torque = np.sum(dQ_arr)
    Power = omega * Torque
    eta = (Thrust * U0 / Power) if Power > 0 else np.nan

    return {
        "mu_arr": mu_arr,
        "r_arr": r_arr,
        "a_arr": a_arr,
        "ap_arr": ap_arr,
        "phi_arr": phi_arr,
        "alpha_arr": alpha_arr,
        "W_arr": W_arr,
        "F_arr": F_arr,
        "dT_arr": dT_arr,
        "dQ_arr": dQ_arr,
        "aoa_in_bounds_arr": aoa_in_bounds_arr,
        "Thrust": Thrust,
        "Torque": Torque,
        "Power": Power,
        "eta": eta
    }


test = False
if test == True:
    collective_pitch = 46
    n_annuli = 40
    results = prop_performance(B, R, R0, U0, omega, collective_pitch, lam, n_annuli, solve_section)

    # -------------------------
    # Print overall performance
    # -------------------------
    print(f"U0       = {U0:.3f} m/s")
    print(f"RPM      = {RPM:.3f}")
    print(f"Thrust   = {results['Thrust']:.3f} N")
    print(f"Torque   = {results['Torque']:.3f} N·m")
    print(f"Power    = {results['Power']:.3f} W")
    print(f"Efficiency = {results['eta']:.5f}")

    # -------------------------
    # Plot lift and drag coefficients
    # -------------------------
    plt.figure()
    plt.plot(alpha_tab, cl_tab, label="Cl")
    plt.plot(alpha_tab, cd_tab, label="Cd")
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Coefficient")
    plt.title("Airfoil Lift and Drag")
    plt.legend()
    plt.grid(True)
    plt.show()

    mu = results["mu_arr"]
    out_b = ~results["aoa_in_bounds_arr"]

    def _mark_out_of_bounds(ys, first_label="AOA out of polar bounds"):
        """Overlay red dots on all given y-arrays at out-of-bounds stations.
        Only the first series gets a legend label."""
        if not out_b.any():
            return
        for i, y in enumerate(ys):
            plt.plot(mu[out_b], y[out_b], 'ro',
                     label=first_label if i == 0 else "_nolegend_")

    # -------------------------
    # Plot induction factors and Prandtl factor
    # -------------------------
    plt.figure()
    plt.plot(mu, results["a_arr"], label="a (axial)")
    plt.plot(mu, results["ap_arr"], label="a' (tangential)")
    plt.plot(mu, results["F_arr"], label="F (Prandtl)")
    _mark_out_of_bounds([results["a_arr"], results["ap_arr"], results["F_arr"]])
    plt.xlabel("r/R")
    plt.ylabel("Coefficient")
    plt.title("Induction Factors along the Blade")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------
    # Plot inflow angle and angle of attack
    # -------------------------
    plt.figure()
    plt.plot(mu, results["phi_arr"], label="phi [deg]")
    plt.plot(mu, results["alpha_arr"], label="alpha [deg]")
    _mark_out_of_bounds([results["phi_arr"], results["alpha_arr"]])
    plt.xlabel("r/R")
    plt.ylabel("Angle [deg]")
    plt.title("Inflow Angle and Angle of Attack")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------
    # Plot differential thrust and torque along radius
    # -------------------------
    plt.figure()
    plt.plot(mu, results["dT_arr"], label="dT (per annulus)")
    plt.plot(mu, results["dQ_arr"], label="dQ (per annulus)")
    _mark_out_of_bounds([results["dT_arr"], results["dQ_arr"]])
    plt.xlabel("r/R")
    plt.ylabel("Force / Torque per annulus")
    plt.title("Differential Thrust and Torque")
    plt.legend()
    plt.grid(True)
    plt.show()

    F_prandtl = np.array([prandtl_tip_root(m, results["a_arr"][i], results["ap_arr"][i])[0] for i, m in enumerate(mu)])
    plt.figure()
    plt.plot(mu, F_prandtl, label="F (Prandtl)")
    _mark_out_of_bounds([F_prandtl])
    plt.xlabel("r/R")
    plt.ylabel("F")
    plt.title("Prandtl's Tip and Root Loss Factor")
    plt.legend()
    plt.grid(True)
    plt.show()
    