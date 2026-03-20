import numpy as np
import matplotlib.pyplot as plt
from standard_regime import *

# -----------------------------
# Turbine (Energy Harvesting) Section Solver
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from standard_regime import *
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
        a_new = - a_from_CT(CT_loc)

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
# Performance sweep for turbine mode
# -----------------------------
def turbine_performance_vs_J(J_array, n_annuli=40):
    eta_arr = []
    P_arr = []
    T_arr = []

    A = np.pi * R**2

    for J in J_array:
        # Update rotation rate from advance ratio
        n_rev = U0 / (J * 2 * R)
        omega = 2.0 * np.pi * n_rev

        # Run BEM with harvesting solver
        results = prop_performance(B, R, R0, U0, omega, lam, n_annuli, solve_section_harvest)

        T = results["Thrust"]
        Q = results["Torque"]
        P = omega * Q   # This will be NEGATIVE for turbine

        # Efficiency definition (energy harvesting)
        eta = -P / (0.5 * rho * U0**3 * A)

        eta_arr.append(eta)
        P_arr.append(P)
        T_arr.append(T)

    return np.array(eta_arr), np.array(P_arr), np.array(T_arr)


# -----------------------------
# Run sweep
# -----------------------------
J_array = np.linspace(2.5, 4.0, 40)

eta_arr, P_arr, T_arr = turbine_performance_vs_J(J_array)

# -----------------------------
# Find optimum (maximum harvested power = most negative P)
# -----------------------------
idx_opt = np.argmin(P_arr)

J_opt = J_array[idx_opt]
P_opt = P_arr[idx_opt]
T_opt = T_arr[idx_opt]
eta_opt = eta_arr[idx_opt]

# -----------------------------
# Print results
# -----------------------------
print("\n--- Optimal Energy Harvesting ---")
print(f"Optimal J           = {J_opt:.3f}")
print(f"Max Power Extracted = {-P_opt:.3f} W")
print(f"Thrust              = {T_opt:.3f} N")
print(f"Energy-harvesting Efficiency = {eta_opt:.5f}")

# -----------------------------
# Plot efficiency vs J
# -----------------------------
plt.figure()
plt.plot(J_array, eta_arr, label="Efficiency")
plt.axvline(J_opt, linestyle="--", label=f"Optimal J = {J_opt:.2f}")
plt.xlabel("Advance Ratio J")
plt.ylabel("Energy-harvesting efficiency η")
plt.title("Energy Harvesting Efficiency vs Advance Ratio")
plt.legend()
plt.grid(True)
plt.show()