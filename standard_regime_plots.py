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

def evaluate_propeller(J, n_annuli=200):
    n_rev = U0 / (J * D_prop)
    RPM = 60.0 * n_rev
    omega = 2.0 * np.pi * n_rev
    lam = omega * R / U0

    mu_root = R0 / R
    mu_arr = np.linspace(mu_root + 1e-3, 0.999, n_annuli)
    r_arr = mu_arr * R
    dr = np.gradient(r_arr)

    dT_arr = np.zeros(n_annuli)
    dQ_arr = np.zeros(n_annuli)

    for i in range(len(mu_arr)-1):
        sol = solve_section(mu_arr[i], mu_arr[i+1], omega=omega,)
        mu = (mu_arr[i]+ mu_arr[i+1]) / 2.0
        r = mu * R
        dT_arr[i] = B * sol["Fax_blade"] * dr[i]
        dQ_arr[i] = B * sol["Ftan_blade"] * r * dr[i]

    Thrust = np.sum(dT_arr)
    Torque = np.sum(dQ_arr)
    Power = omega * Torque

    CT = Thrust / (rho * n_rev**2 * D_prop**4)
    CP = Power / (rho * n_rev**3 * D_prop**5)
    eta = (J * CT / CP) if CP > 0 else np.nan

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
        "eta": eta,
    }

J_vals = np.linspace(0.1, 2.7, 40)

CT_vals = []
CP_vals = []
eta_vals = []

for J in J_vals:
    res = evaluate_propeller(J)
    CT_vals.append(res["CT"])
    CP_vals.append(res["CP"])
    eta_vals.append(res["eta"])
    print(f"J={J:.3f}, CT={res['CT']:.5f}, CP={res['CP']:.5f}, eta={res['eta']:.5f}")

CT_vals = np.array(CT_vals)
CP_vals = np.array(CP_vals)
eta_vals = np.array(eta_vals)


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
                eta = float(cols[7]) / 100.0  # convert % to 0-1
                data.append([J, CT, CP, eta])
            except ValueError:
                continue

    data = np.array(data)
    return {
        'J': data[:,0],
        'CT': data[:,1],
        'CP': data[:,2],
        'eta': data[:,3]
    }


# ===============================
# 2️⃣ Example: Plot my data vs JavaFoil
# ===============================

# Load JavaFoil results
jf_data = load_javafoil_data("javaprop_validation_plots.txt")

# Assume your computed data arrays:
# J_vals, CT_vals, CP_vals, eta_vals

plt.figure()
plt.plot(J_vals, CT_vals, 'o-', label='My BEM')
plt.plot(jf_data['J'], jf_data['CT'], 's--', label='JavaFoil')
plt.xlabel("Advance ratio J")
plt.ylabel(r"$C_T$")
plt.title("Thrust Coefficient vs Advance Ratio")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(J_vals, CP_vals, 'o-', label='My BEM')
plt.plot(jf_data['J'], jf_data['CP'], 's--', label='JavaFoil')
plt.xlabel("Advance ratio J")
plt.ylabel(r"$C_P$")
plt.title("Power Coefficient vs Advance Ratio")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(J_vals, eta_vals, 'o-', label='My BEM')
plt.plot(jf_data['J'], jf_data['eta'], 's--', label='JavaFoil')
plt.xlabel("Advance ratio J")
plt.ylabel(r"$\eta$")
plt.title("Propeller Efficiency vs Advance Ratio")
plt.legend()
plt.grid(True)

plt.show()