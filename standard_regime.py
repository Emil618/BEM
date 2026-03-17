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
    return np.interp(alpha_deg, alpha_tab, cl_tab)

def Cd(alpha_deg):
    return np.interp(alpha_deg, alpha_tab, cd_tab)


# Defining functions describing blade geometry

def beta(mu): # mu = r/R
    return np.radians(-50.0 * mu + 81.0)

def chord(mu): # mu = r/R
    return R * (0.18 - 0.06 * mu)

# Prandtl's correction for finite number of blades, including tip and root losses

def prandtl_tip_root(mu, a, ap): # mu = r/R (radial position), a = axial induction factor, ap = tangential induction factor
    mu_root = R0 / R
    mu = np.clip(mu, mu_root + 1e-8, 1.0 - 1e-8)

    denom = max(1.0 + a, 1e-8)
    sqrt_term = np.sqrt(1.0 + ((lam * mu) / denom) ** 2)

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


def solve_section(mu1,mu2,omega, a0=0.3, ap0=0.01, max_iter=500, tol=1e-6, relax=0.1): 

    """
    This function iteratively computes the axial and tangential induction factors (a, a') for a 
    radial annulus of a propeller, taking into account Prandtl's tip and root loss corrections 
    and aerodynamic forces from lift and drag coefficients. It returns the local velocities, 
    inflow angles, forces, and correction factors for the section.

    Parameters:
        mu1 : float
            Non-dimensional radial position at the inner edge of the annulus (r/R).
        mu2 : float
            Non-dimensional radial position at the outer edge of the annulus (r/R).
        a0 : float, optional
            Initial guess for axial induction factor (default: 0.3).
        ap0 : float, optional
            Initial guess for tangential induction factor (default: 0.01).
        max_iter : int, optional
            Maximum number of iterations for convergence (default: 500).
        tol : float, optional
            Convergence tolerance for induction factors (default: 1e-6).
        relax : float, optional
            Relaxation factor for iterative updates of a and a' (default: 0.25).

    Returns:
        dict :
            Dictionary containing the following outputs for the blade section:
            - "a" : float
                Axial induction factor.
            - "ap" : float
                Tangential induction factor.
            - "phi" : float
                Inflow angle at the section [rad].
            - "alpha" : float
                Angle of attack at the section [rad].
            - "W" : float
                Relative wind speed at the section [m/s].
            - "Fax_blade" : float
                Axial force per blade at the section [N].
            - "Ftan_blade" : float
                Tangential force per blade at the section [N].
            - "F" : float
                Prandtl tip/root loss correction factor.
            - "ftip" : float
                Tip loss factor.
            - "froot" : float
                Root loss factor.
    """

    mu = (mu1 + mu2) / 2.0 # Midpoint of the annulus for calculations
    Area = np.pi * R**2 * (mu2**2 - mu1**2) # Area of the annulus corresponding to the blade element
    r = mu * R #Radial position of the blade element in meters
    c_local = chord(mu) #Chord length at the blade element in meters
    sigma = B * c_local / (2.0 * np.pi * r) #Local solidity of the blade element

    # Initialize induction factors
    a = a0
    ap = ap0
    converged = False

    # Iteratively solve for a and ap using the blade element momentum theory
    for _ in range(max_iter):
        Vax = U0 * (1.0 + a) #Axial velocity at the blade section
        Vtan = omega * r * (1.0 - ap) #Tangential velocity at the blade section

        Vtan = max(Vtan, 1e-8)
        phi = np.arctan2(Vax, Vtan) #Inflow angle at the blade section

        s = np.sin(phi)
        c = np.cos(phi)
        s2 = max(s * s, 1e-10)
        sc = np.sign(s * c) * max(abs(s * c), 1e-10)

        alpha = beta(mu) - phi #Angle of attack at the blade section based on blade geometry and inflow angle
        alpha_deg = np.degrees(alpha)

        cl = Cl(alpha_deg) # Lift coefficient at the blade section based on angle of attack
        cd = Cd(alpha_deg) # Drag coefficient at the blade section based on angle of attack

        Cn = cl * c + cd * s # Normal force coefficient at the blade section
        Ct = cl * s - cd * c # Tangential force coefficient at the blade section

        F, _, _ = prandtl_tip_root(mu, a, ap) # Calculating Prandtl's tip and root loss correction factor

        # kx = sigma * Cn / (4.0 * F * s2) # Intermediate variable for axial induction factor update based on momentum theory
        ky = sigma * Ct / (4.0 * F * sc) # Intermediate variable for tangential induction factor update based on momentum theory

        # a_new = kx / max(1.0 - kx, 1e-8) # Update axial induction factor based on momentum theory nad linearization of root finding
        ap_new = ky / (1.0 + ky) # Update tangential induction factor based on momentum theory and linearization of root finding

        # Local thrust coefficient (annulus form)
        CT_loc = sigma * Cn / (F * s2)

        # Convert CT → a using Glauert correction
        a_new = a_from_CT(CT_loc)

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


def prop_performance(B, R, R0, U0, omega, lam, n_annuli, solve_section):
    """
    Compute propeller performance using BEM (propeller-style convention).

    Parameters:
        B           : int       - number of blades
        R           : float     - propeller radius [m]
        R0          : float     - hub radius [m]
        U0          : float     - inflow velocity [m/s]
        omega       : float     - rotation rate [rad/s]
        n_annuli    : int       - number of blade elements

    Returns:
        dict with arrays: mu_arr, a_arr, ap_arr, phi_arr, alpha_arr, W_arr, F_arr,
                          dT_arr, dQ_arr, r_arr, Thrust, Torque, Power, eta
    """
    mu_arr = np.linspace(R0 / R, 1.0, n_annuli)
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

    for i in range(n_annuli - 1):
        sol = solve_section(mu_arr[i], mu_arr[i+1], omega, lam)  # BEM solution for annulus
        mu = (mu_arr[i] + mu_arr[i+1]) / 2.0  # midpoint for storing results
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
        "Thrust": Thrust,
        "Torque": Torque,
        "Power": Power,
        "eta": eta
    }


test = True
if test == True:
    n_annuli = 200
    results = prop_performance(B, R, R0, U0, omega, lam, n_annuli, solve_section)

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

    # -------------------------
    # Plot induction factors and Prandtl factor
    # -------------------------
    plt.figure()
    plt.plot(results["mu_arr"], results["a_arr"], label="a (axial)")
    plt.plot(results["mu_arr"], results["ap_arr"], label="a' (tangential)")
    plt.plot(results["mu_arr"], results["F_arr"], label="F (Prandtl)")
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
    plt.plot(results["mu_arr"], results["phi_arr"], label="phi [deg]")
    plt.plot(results["mu_arr"], results["alpha_arr"], label="alpha [deg]")
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
    plt.plot(results["mu_arr"], results["dT_arr"], label="dT (per annulus)")
    plt.plot(results["mu_arr"], results["dQ_arr"], label="dQ (per annulus)")
    plt.xlabel("r/R")
    plt.ylabel("Force / Torque per annulus")
    plt.title("Differential Thrust and Torque")
    plt.legend()
    plt.grid(True)
    plt.show()