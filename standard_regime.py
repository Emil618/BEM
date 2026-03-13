import numpy as np
import matplotlib.pyplot as plt

R = 0.7
B = 6
h = 2000
R0 = 0.25 * R
U0 = 60.0
J = 2.0

rho = 1.225 * (1 - 2.25577e-5 * h) ** 4.256

D_prop = 2 * R
n_rev = U0 / (J * D_prop)
RPM = 60.0 * n_rev
omega = 2.0 * np.pi * n_rev
lam = omega * R / U0

data = np.loadtxt("ARAD8pct_polar.txt", comments="#")
alpha_tab = data[:, 0]
cl_tab = data[:, 1]
cd_tab = data[:, 2]

def Cl(alpha_deg):
    return np.interp(alpha_deg, alpha_tab, cl_tab)

def Cd(alpha_deg):
    return np.interp(alpha_deg, alpha_tab, cd_tab)

def beta(mu):
    return np.radians(-50.0 * mu + 81.0)

def chord(mu):
    return R * (0.18 - 0.06 * mu)

def prandtl_tip_root(mu, a, ap):
    mu_root = R0 / R
    mu = np.clip(mu, mu_root + 1e-8, 1.0 - 1e-8)

    denom = max(1.0 + a, 1e-8)
    sqrt_term = np.sqrt(1.0 + ((lam * mu * (1.0 - ap)) / denom) ** 2)

    expo_tip = -(B / 2.0) * ((1.0 - mu) / mu) * sqrt_term
    expo_root = -(B / 2.0) * ((mu - mu_root) / mu) * sqrt_term

    f_tip_arg = np.exp(np.clip(expo_tip, -700.0, 0.0))
    f_root_arg = np.exp(np.clip(expo_root, -700.0, 0.0))

    f_tip_arg = np.clip(f_tip_arg, 0.0, 1.0)
    f_root_arg = np.clip(f_root_arg, 0.0, 1.0)

    f_tip = (2.0 / np.pi) * np.arccos(f_tip_arg)
    f_root = (2.0 / np.pi) * np.arccos(f_root_arg)

    F = f_tip * f_root
    return np.clip(F, 1e-4, 1.0), f_tip, f_root

def solve_section(mu, a0=0.05, ap0=0.01, max_iter=500, tol=1e-6, relax=0.25):
    r = mu * R
    c_local = chord(mu)
    sigma = B * c_local / (2.0 * np.pi * r)

    a = a0
    ap = ap0
    converged = False

    for _ in range(max_iter):
        Vax = U0 * (1.0 + a)
        Vtan = omega * r * (1.0 - ap)

        Vtan = max(Vtan, 1e-8)
        phi = np.arctan2(Vax, Vtan)

        s = np.sin(phi)
        c = np.cos(phi)
        s2 = max(s * s, 1e-10)
        sc = np.sign(s * c) * max(abs(s * c), 1e-10)

        alpha = beta(mu) - phi
        alpha_deg = np.degrees(alpha)

        cl = Cl(alpha_deg)
        cd = Cd(alpha_deg)

        Cn = cl * c - cd * s
        Ct = cl * s + cd * c

        F, _, _ = prandtl_tip_root(mu, a, ap)

        kx = sigma * Cn / (4.0 * F * s2)
        ky = sigma * Ct / (4.0 * F * sc)

        a_new = kx / max(1.0 - kx, 1e-8)
        ap_new = ky / (1.0 + ky)

        a_new = np.clip(a_new, -0.2, 3.0)
        ap_new = np.clip(ap_new, -1.0, 1.0)

        if abs(a_new - a) < tol and abs(ap_new - ap) < tol:
            a = a_new
            ap = ap_new
            converged = True
            break

        a = (1.0 - relax) * a + relax * a_new
        ap = (1.0 - relax) * ap + relax * ap_new

    if not converged:
        print(f"Warning: section at mu={mu:.4f} did not converge")

    Vax = U0 * (1.0 + a)
    Vtan = omega * r * (1.0 - ap)
    phi = np.arctan2(Vax, max(Vtan, 1e-8))
    alpha = beta(mu) - phi
    alpha_deg = np.degrees(alpha)

    cl = Cl(alpha_deg)
    cd = Cd(alpha_deg)

    W = np.hypot(Vax, Vtan)
    q = 0.5 * rho * W**2

    L = q * cl * c_local
    D = q * cd * c_local

    dF_ax_per_blade = L * np.cos(phi) - D * np.sin(phi)
    dF_tan_per_blade = L * np.sin(phi) + D * np.cos(phi)

    F, ftip, froot = prandtl_tip_root(mu, a, ap)

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
    }

n_annuli = 60
mu_root = R0 / R
mu_arr = np.linspace(mu_root + 1e-3, 0.999, n_annuli)
r_arr = mu_arr * R
dr = np.gradient(r_arr)

a_arr = np.zeros(n_annuli)
ap_arr = np.zeros(n_annuli)
phi_arr = np.zeros(n_annuli)
alpha_arr = np.zeros(n_annuli)
W_arr = np.zeros(n_annuli)
F_arr = np.zeros(n_annuli)

dT_arr = np.zeros(n_annuli)
dQ_arr = np.zeros(n_annuli)

for i, mu in enumerate(mu_arr):
    sol = solve_section(mu)

    r = mu * R
    a_arr[i] = sol["a"]
    ap_arr[i] = sol["ap"]
    phi_arr[i] = np.degrees(sol["phi"])
    alpha_arr[i] = np.degrees(sol["alpha"])
    W_arr[i] = sol["W"]
    F_arr[i] = sol["F"]

    dT_arr[i] = B * sol["Fax_blade"] * dr[i]
    dQ_arr[i] = B * sol["Ftan_blade"] * r * dr[i]

Thrust = np.sum(dT_arr)
Torque = np.sum(dQ_arr)
Power = omega * Torque
eta = (Thrust * U0 / Power) if Power > 0 else np.nan

print(f"U0     = {U0:.3f} m/s")
print(f"J      = {J:.3f}")
print(f"RPM    = {RPM:.3f}")
print(f"lambda = {lam:.3f}")
print(f"Thrust = {Thrust:.3f} N")
print(f"Torque = {Torque:.3f} N m")
print(f"Power  = {Power:.3f} W")
print(f"eta    = {eta:.5f}")

plt.figure()
plt.plot(alpha_tab, cl_tab, label="Cl")
plt.plot(alpha_tab, cd_tab, label="Cd")
plt.xlabel("alpha [deg]")
plt.ylabel("Coefficient")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(mu_arr, a_arr, label="a")
plt.plot(mu_arr, ap_arr, label="a'")
plt.plot(mu_arr, F_arr, label="F")
plt.xlabel("r/R")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(mu_arr, phi_arr, label="phi [deg]")
plt.plot(mu_arr, alpha_arr, label="alpha [deg]")
plt.xlabel("r/R")
plt.legend()
plt.grid(True)
plt.show()