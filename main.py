import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import logging

from vehicle import VehicleModel
from beam import Beam
from load_els import LoadElement, LoadSystem
import fem_utils
from data_utils import get_err, interpolate_mat, get_fft
from bridge_profile import generate_harmonic_profile
from solver import solve_system, get_fade_weights
import modal

abaqus_cmd = r"C:\SIMULIA\Commands\abaqus.bat"
verify_mat = False

g = 9.81

## Input

# Vehicle data
vel = 20

l_veh = 2
h_veh = 0.5

m_carriage = 3e3
m_axles = [3e2, 3e2]
ms = np.array([m_carriage] + m_axles) # Vehicle weights
mtot = sum(ms)
mw = np.array([0, 0, 0])

if len(mw) != len(m_axles):
    raise ValueError("Length of mw must be equal to length of m_axles")

Iz = 1/12*m_carriage*(l_veh**2 + h_veh**2)
J_rot = [Iz]

cs = np.array([[0, 0], [0, 0]])
ks = np.array([[2e4, 2e4], [2e3, 2e3]]) # Spring stiffnesses
ks_last = ks[-1]
cs_last = cs[-1]
num_axles = ks.shape[1]

vehicle = VehicleModel(ms, cs, ks, l_veh, J_rot, include_pitch=True)
vehicle.plot_modes()
vehicle.plot_TF(50, 1)
n_dof_vehicle = vehicle.M.shape[0]
wn_v = np.sqrt(vehicle.eigenvals)
np.savetxt('vehicle_freqs.txt', wn_v, fmt='%.3f')

# configuration for moving load
loads = []
loads.append(LoadElement(np.array(m_axles), m_carriage, np.array([l_veh])))
load_configuration = LoadSystem(loads, np.array([]))

startdofcontact = len(J_rot) + 1

# Beam data
length_b = 25
mu = 3000
E = 3.1e10
h = 3
b = 11
J = h**3*b/12
damping_ratios = [0.01, 1e-5]

omega = np.pi * vel/length_b

n_modes_b = 2

# Abaqus parameters
num_nodes = 1001
x = np.linspace(0, length_b, num_nodes)
dx = length_b / (num_nodes - 1)
max_freq_abq = (n_modes_b*np.pi/length_b)**2*np.sqrt(E*J/mu)

from_FEM = False

parameters_fem = {'height': h,
                  'base': b,
                  'density': mu,
                  'max_freq': max_freq_abq,
                  'n_modes': n_modes_b,
                  'n_nodes': num_nodes,
                  'dx': dx}

base_path = 'modal_template'
out_inp_path = 'modal'

if from_FEM:
    fem_utils.write_input(parameters_fem, base_path, out_inp_path)
    fem_utils.run_job(out_inp_path, abaqus_cmd)

logging.basicConfig(
    filename='modalcoupled.log',       # Log file name
    level=logging.INFO,             # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'
)
separator = "=" * 60
logging.info(f"\n{separator}")

# 1. Load data
U2_modes, freqs = modal.get_modes(E, J, mu, x, n_modes_b, from_FEM)
wn_b = 2 * np.pi * freqs

with open('bridge_freqs.txt', 'w') as file:
    file.write(f"Omega_v: {omega:.3f}\n\n")
    for freq in wn_b:
        file.write(f"{freq:.3f}\n")

logging.info(f'Analysis started with {n_modes_b} modes')

xc0 = np.array([0] + [-l_veh] * (num_axles - 1))
T = (np.max(x)+np.max(np.abs(xc0)))/vel
fmax = 2000
dt = 1/fmax
time = np.arange(0, T, dt)
n_steps = len(time)

logging.info(f'MR = {mtot/(mu*np.max(x))}')

# State vectors for bridge
y_b = np.zeros((n_steps, n_modes_b))
y_b_dot = np.zeros((n_steps, n_modes_b))

# Lagrangian variables for vehicle
x_v = np.zeros((n_dof_vehicle, time.shape[0]))
xdot_v = np.zeros_like(x_v)
xddot_v = np.zeros_like(x_v)

U2modes_interp = CubicSpline(x, U2_modes, axis=0)

# y_b and y_b_dot: modal coordinates
y_b = np.zeros((n_steps, n_modes_b))
y_b_dot = np.zeros((n_steps, n_modes_b))

force_contact = np.empty((num_axles, len(time)))
force_contact[:] = np.nan
force_contact[0,0] = -(ms[1] + m_carriage/num_axles) * g

U2 = np.zeros((U2_modes.shape[0], len(time)))
U2_contacts = np.zeros((num_axles, len(time)))

f0_spatial = 2*wn_v[0]/(2*np.pi)/vel
r_interp = generate_harmonic_profile(np.max(x), len(x), f0_spatial, 0)
dr_dx_interp = r_interp.derivative()

alphaR, betaR = modal.get_rayleigh_pars(wn_b, damping_ratios)

nx_beam = 400
nt_beam = 500

my_beam = Beam(np.max(x), mu, E, J, 0, n_modes_b, nx_beam, nt_beam, vel)
t_global, v, bm = my_beam.compute_multi_response(load_configuration)

plt.figure()
plt.plot(t_global, v[v.shape[0]//2], label='Analytical')
plt.show()

def modal_to_natural_global(yb: np.ndarray):
    U2 = U2_modes @ yb
    return U2

def modal_to_natural_contact(U2modes_contact: np.ndarray, yb: np.ndarray):
    U2_contact = U2modes_contact @ yb.reshape(-1,1)
    return U2_contact

def get_contact_car(U2modes_contact, yb: np.ndarray, x_v: np.ndarray, r_c: np.ndarray):
    U2_contact = modal_to_natural_contact(U2modes_contact, yb).squeeze()
    f = ks_contact * (x_v - U2_contact - r_c)
    return f

U = np.zeros(n_modes_b + n_dof_vehicle)
dotU = np.zeros_like(U)
ddotU = np.zeros_like(U)

for i in range(1, n_steps):
    t = time[i]

    x_c = xc0 + vel * t
    mask = (x_c >= 0) & (x_c <= length_b)
    x_c = x_c[mask]
    idx_inside = np.where(mask)[0]
    dof_inside = startdofcontact + idx_inside

    ks_contact = ks_last[idx_inside]
    cs_contact = cs_last[idx_inside]

    ramp_duration = 3*dt
    apply_fade = False

    fade_weights = get_fade_weights(ramp_duration, apply_fade, idx_inside, xc0, vel, t, length_b)

    if not apply_fade:
        fade_weights[:] = 1

    r_c = r_interp(x_c)
    dr_dx_c = dr_dx_interp(x_c)

    U0 = U
    dotU0 = dotU
    ddotU0 = ddotU

    U2modes_contact = U2modes_interp(x_c).reshape(len(dof_inside),-1)

    config = {
        "n_modes_b": n_modes_b,
        "n_modes_v": n_dof_vehicle,

        "U2modes_contact": U2modes_contact,  # shape: (num_contact_dofs, n_modes_b)

        # Contact parameters (per axle)
        "ks_contact": ks_contact,            # shape: (num_contact_dofs,)
        "cs_contact": cs_contact,            # shape: (num_contact_dofs,)

        # Vehicle matrices
        "Mv": vehicle.M,                            # shape: (n_modes_v, n_modes_v)
        "Cv": vehicle.C,                            # shape: (n_modes_v, n_modes_v)
        "Kv": vehicle.K,                            # shape: (n_modes_v, n_modes_v)

        # Bridge modal properties
        "wn_b": wn_b,           # shape: (n_modes_b,)

        # Rayleigh damping parameters
        "alphaR": alphaR,
        "betaR": betaR,

        # Mass information
        "ms": ms,                            # shape: (n_vehicle_dofs,)
        "m_carriage": m_carriage,
        "num_axles": num_axles,
        "mw": mw,

        # Contact state (which axles are on bridge)
        "idx_inside": idx_inside,           # shape: (num_contact_dofs,)

        # Vehicle velocity
        "vel": vel,                          # scalar or array per axle (depending on your implementation)

        # Gravity and time step
        "g": g,
        "dt": dt,

        # Local roughness data
        "r_c": r_c,
        "dr_dx_c": dr_dx_c,

        # interaction vehicle dofs
        "dof_inside": dof_inside,

        # fade data
        "fade_weights": fade_weights
    }

    U, dotU, ddotU = solve_system(U0, dotU0, ddotU0, config)

    ybt = U[:n_modes_b]
    ydot_bt = dotU[:n_modes_b]

    xv_t = U[n_modes_b:]
    xdot_vt = dotU[n_modes_b:]
    xddot_vt = ddotU[n_modes_b:]

    U2_t = modal_to_natural_global(ybt)

    U2_contact = np.empty(num_axles)
    U2_contact[:] = np.nan
    U2_contact[idx_inside] = modal_to_natural_contact(U2modes_contact, ybt).squeeze()

    U2[:, i] = U2_t
    U2_contacts[:,i] = U2_contact

    x_v[:,i] = xv_t
    xdot_v[:,i] = xdot_vt
    xddot_v[:,i] = xddot_vt

    y_b[i] = ybt
    y_b_dot[i] = ydot_bt

    fsi_contact = np.empty(num_axles)
    fsi_contact[:] = np.nan
    fsi_contact[idx_inside] = get_contact_car(U2modes_contact, ybt, xv_t[dof_inside], r_c)

    mass_per_axle = np.array(m_axles) + m_carriage/num_axles

    fsi_contact += - mass_per_axle*g
    force_contact[:,i] = fsi_contact

# Plots
idx_axle_plot = 1
logging.info('ERRORS')
plt.figure()
y = force_contact[idx_axle_plot]
plt.plot(time, y)
plt.xlabel(r'$t$')
plt.ylabel(r'$F_c$')
plt.title('Contact force')
plt.show()

dofv = 1
dofv_matlab = 0

plt.figure('Vehicle')
plt.subplot(3,1,1)
plt.plot(time, x_v[dofv])
plt.xlabel(r'$t$')
plt.ylabel(r'$x_v$')
plt.title('Displacement')

plt.subplot(3,1,2)
plt.plot(time, xdot_v[dofv])
plt.xlabel(r'$t$')
plt.ylabel(r'$\dot{x}_v$')
plt.title('Speed')

plt.subplot(3,1,3)
plt.plot(time, xddot_v[dofv])
plt.xlabel(r'$t$')
plt.ylabel(r'$\dot{x}_v$')
plt.title('Acceleration')

plt.tight_layout()
plt.show()

plt.plot(time, U2_contacts[idx_axle_plot])
plt.xlabel(r'$t$')
plt.ylabel(r'$x_c$')
plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.title('Contact point displacement')
plt.show()

vel_contact = np.gradient(U2_contacts[idx_axle_plot], time)
acc_contact = np.gradient(vel_contact, time)

plt.plot(time, acc_contact)
plt.xlabel(r'$t$')
plt.ylabel(r'$a_c$')
plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.title('Contact point accleration')
plt.show()

U2_noint_mid_interp = interp1d(t_global, -v[v.shape[0]//2])
U2_noint_mid = U2_noint_mid_interp(time)

U2_mid = U2[U2.shape[0]//2]

diff_noint_int = get_err(U2_noint_mid, U2_mid)
logging.info(f'DIFF INT_NOINT = {diff_noint_int}')

plt.figure(dpi=300)
plt.plot(time, U2_mid, label='interaction')
plt.plot(time, U2_noint_mid, label='no interaction')
plt.xlabel(r'$t$')
plt.ylabel(r'$U_2$')
plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.title('Midspan displacement')
plt.legend()
plt.show()

# Verification

# exclude eventual nans from contact force
mask=~np.isnan(force_contact[idx_axle_plot])
x_v = x_v[dofv]

if verify_mat:
    dict_matlab = loadmat('VerificationVBI.mat')
    U2_mat = dict_matlab['U_xt'].squeeze()
    t_mat = dict_matlab['t_ver'].squeeze()
    F_contact_mat = dict_matlab['F_contact'].squeeze()
    midnode = dict_matlab['node_midspan'].item()
    U2mid_mat = U2_mat[midnode]
    x_v_mat = dict_matlab['U_veh'].squeeze()

    U2mid_mat = interpolate_mat(t_mat, U2mid_mat, time)
    err = get_err(U2mid_mat, U2_mid)
    logging.info(f'ERROR MATLAB - U2 MIDSPAN = {err}')

    plt.figure()
    plt.plot(time, U2mid_mat, label='Matlab')
    plt.plot(time, U2_mid, label='Python')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$U_2$')
    plt.legend()
    plt.title('Midspan displacement verification')
    plt.show()

    x_v_mat = interpolate_mat(t_mat, x_v_mat, time)
    x_v_mat = x_v_mat[dofv_matlab]

    err = get_err(x_v, x_v_mat)
    logging.info(f'ERROR MATLAB - VEHICLE = {err}')
    plt.figure()
    plt.plot(time, x_v_mat, label='Matlab')
    plt.plot(time, x_v, label='Python')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x_v$')
    plt.title(f'DOF {dofv} displacement verification')
    plt.legend()
    plt.show()

    # Contact force
    F_contact_mat = interpolate_mat(t_mat, F_contact_mat[idx_axle_plot], time)
    err = get_err(F_contact_mat[mask], force_contact[idx_axle_plot, mask])
    logging.info(f'ERROR MATLAB - CONTACT FORCE = {err}')

    plt.figure()
    plt.plot(time, F_contact_mat, label='Matlab')
    plt.plot(time, force_contact[idx_axle_plot], label='Python')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$F$')
    plt.title('Contact force verification')
    plt.legend()
    plt.show()

## Data analysis
get_fft(time[mask], y[mask], f'FFT contact force axle {idx_axle_plot}')
get_fft(time, xddot_v[dofv], f'FFT acceleration DOF {dofv}')
get_fft(time, x_v, f'FFT displacement DOF {dofv}')
