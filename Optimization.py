import numpy as np
import matplotlib.pyplot as plt
from standard_regime import *
from scipy.optimize import differential_evolution

# -----------------------------
# Turbine (Energy Harvesting) Section Solver
# -----------------------------
def solve_section_harvest(mu1, mu2, omega, a0=0.1, ap0=0.01, max_iter=500, tol=1e-6, relax=0.1):
    mu = (mu1 + mu2) / 2.0
    r = mu * R
    c_local = chord(mu)
    sigma = B * c_local / (2.0 * np.pi * r)

    a = -abs(a0)
    ap = -abs(ap0)

    for _ in range(max_iter):
        Vax = U0 * (1.0 - a)
        Vtan = omega * r * (1.0 + ap)
        phi = np.arctan2(Vax, Vtan)
        s, c = np.sin(phi), np.cos(phi)
        sc, s2 = s * c, s * s
        alpha = beta(mu) - phi
        cl, cd = Cl(np.degrees(alpha)), Cd(np.degrees(alpha))
        Cn, Ct = cl*c - cd*s, cl*s + cd*c
        F, _, _ = prandtl_tip_root(mu, a, ap)

        ky = max(sigma * Ct / (4.0 * F * sc), 1e-8)
        ap_new = -(-1.0 + np.sqrt(1.0 + 4.0 * ky)) / 2.0
        CT_loc = sigma * Cn / (F * s2)
        a_new = -a_from_CT(CT_loc)

        a = 0.9*a + 0.1*a_new
        ap = 0.9*ap + 0.1*ap_new

    Vax = U0 * (1 - a)
    Vtan = omega * r * (1 + ap)
    phi = np.arctan2(Vax, Vtan)
    alpha = beta(mu) - phi
    cl, cd = Cl(np.degrees(alpha)), Cd(np.degrees(alpha))
    W = np.hypot(Vax, Vtan)
    q = 0.5 * rho * W**2
    L, D = q*cl*c_local, q*cd*c_local
    Fax_blade = L*np.cos(phi) - D*np.sin(phi)
    Ftan_blade = L*np.sin(phi) + D*np.cos(phi)
    F, ftip, froot = prandtl_tip_root(mu, a, ap)

    return {
        "a": a,
        "ap": ap,
        "phi": phi,
        "alpha": alpha,
        "W": W,
        "Fax_blade": Fax_blade,
        "Ftan_blade": Ftan_blade,
        "F": F,
        "ftip": ftip,
        "froot": froot,
    }

# -----------------------------
# Linear blade geometry functions
# -----------------------------
def set_linear_geometry(params):
    chord_root, chord_tip, twist_root, twist_tip, delta_beta = params
    global chord, beta
    chord = lambda mu: np.interp(mu, [R0/R, 1.0], [chord_root, chord_tip])
    beta = lambda mu: np.interp(mu, [R0/R, 1.0], [twist_root, twist_tip]) + np.radians(delta_beta)

# -----------------------------
# Objective function
# -----------------------------
def objective(params):
    set_linear_geometry(params)
    omega_min = 0.5 * U0 / R
    omega_max = 3.0 * U0 / R
    n_annuli = 20
    max_P = -np.inf
    for omega_try in np.linspace(omega_min, omega_max, 10):
        lam_try = omega_try*R/U0
        results = prop_performance(B, R, R0, U0, omega_try, lam_try, n_annuli, solve_section_harvest)
        P_harvest = -results["Power"]
        if P_harvest > max_P:
            max_P = P_harvest
    return -max_P  # negative because we minimize

# -----------------------------
# Progress callback
# -----------------------------
def progress_callback(xk, convergence=0):
    print(f"Current parameter guess: {xk}, convergence: {convergence:.3f}")

# -----------------------------
# Bounds for 5 parameters
# -----------------------------
bounds = [
    (0.05*R, 0.3*R),           # chord_root
    (0.05*R, 0.3*R),           # chord_tip
    (np.radians(0), np.radians(90)),    # twist_root
    (np.radians(-30), np.radians(60)),  # twist_tip
    (-10, 10),                  # delta_beta in deg
]

# -----------------------------
# Run optimizer
# -----------------------------
print("Starting optimization...")
result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=15, popsize=15, 
                                polish=True, callback=progress_callback)
optimal_params = result.x

print("=== Optimal Blade Geometry ===")
print(f"chord_root: {optimal_params[0]:.3f} m")
print(f"chord_tip: {optimal_params[1]:.3f} m")
print(f"twist_root: {np.degrees(optimal_params[2]):.1f} deg")
print(f"twist_tip: {np.degrees(optimal_params[3]):.1f} deg")
print(f"delta_beta: {optimal_params[4]:.1f} deg")

# -----------------------------
# Compute final performance at optimal geometry
# -----------------------------
set_linear_geometry(optimal_params)
omega_min, omega_max = 0.5*U0/R, 3.0*U0/R
omega_vals = np.linspace(omega_min, omega_max, 20)
max_P = -np.inf
best_results = None
best_omega = omega_min
for i, omega_try in enumerate(omega_vals, 1):
    lam_try = omega_try*R/U0
    results = prop_performance(B, R, R0, U0, omega_try, lam_try, 20, solve_section_harvest)
    P_harvest = -results["Power"]
    if P_harvest > max_P:
        max_P = P_harvest
        best_results = results
        best_omega = omega_try
    print(f"[Omega sweep {i}/{len(omega_vals)}] omega={omega_try:.2f} rad/s, Power={P_harvest:.1f} W")

Cp_opt = -best_results["Power"] / (0.5 * rho * U0**3 * np.pi*R**2)

print(f"Optimal omega (rad/s): {best_omega:.2f}")
print(f"Power coefficient Cp: {Cp_opt:.3f}")
print(f"Power harvested (W): {max_P:.1f}")

# -----------------------------
# Plot induction factors
# -----------------------------
plt.figure()
plt.plot(best_results["mu_arr"], best_results["a_arr"], label="a (axial)")
plt.plot(best_results["mu_arr"], best_results["ap_arr"], label="a' (tangential)")
plt.plot(best_results["mu_arr"], best_results["F_arr"], label="F (Prandtl)")
plt.xlabel("r/R")
plt.ylabel("Coefficient")
plt.title("Optimized Energy Harvesting: Induction Factors")
plt.legend()
plt.grid(True)
plt.show()