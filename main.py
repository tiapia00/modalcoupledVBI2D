import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pandas as pd
import logging

import vehicle
from beam import Beam
from load_els import LoadElement, LoadSystem
import fem_utils
from comparison_utils import get_err, interpolate_mat
from profile import generate_harmonic_profile
import modal

abaqus_cmd = r"C:\SIMULIA\Commands\abaqus.bat"

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

Iz = 1/12*m_carriage*(l_veh**2 + h_veh**2)
I = [Iz]

cs = np.array([[0, 0], [0, 0]])
ks = np.array([[2e6, 2e6], [2e5, 2e5]]) # Spring stiffnesses
ks_last = ks[-1]
cs_last = cs[-1]
num_axles = ks.shape[1]

#matrices = vehicle_models.quarter_car(ms, cs, ks)
matrices = vehicle.quarter_car_pitch(ms, I, cs, ks, l_veh)
Mv, Cv, Kv = matrices
n_modes_v = Mv.shape[0]
f_ext_v = -g * np.insert(ms, 0, 0)
xv_eq = np.linalg.solve(Kv, f_ext_v)

try:
    eigenvals_v, modes_v = eigh(Kv, Mv)
except ValueError as e:
    raise RuntimeError(f'Error computing eigenvalues: {e}')

if np.any(eigenvals_v<0):
    raise ValueError('Negative eigenvalue of car: system not stable')

wn_v = np.sqrt(eigenvals_v)
vehicle.plot_modes(modes_v, wn_v, xv_eq, l_veh)

# configuration for moving load
vehs = []
vehs.append(LoadElement(np.array(m_axles), m_carriage, np.array([l_veh])))
load_configuration = LoadSystem(vehs, np.array([]))

startdofcontact = len(I) + 1

# Beam data
length_b = 25
mu = 1000
E = 3.1e10
h = 3
b = 11
J = h**3*b/12
damping_ratios = [0, 0]

omega = np.pi * vel/length_b
print(omega)

n_modes_b = 30

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
circ_freqs = 2 * np.pi * freqs

logging.info(f'Analysis started with {n_modes_b} modes')

xc0 = np.array([0] + [-l_veh] * (num_axles - 1))
T = (np.max(x)+np.max(np.abs(xc0)))/vel
fmax = 20000
dt = 1/fmax
time = np.arange(0, T, dt)
n_steps = len(time)

logging.info(f'MR = {mtot/(mu*np.max(x))}')

# State vectors for bridge
y_b = np.zeros((n_steps, n_modes_b))
y_b_dot = np.zeros((n_steps, n_modes_b))

# Lagrangian variables for vehicle
x_v = np.zeros((n_modes_v, time.shape[0]))
xdot_v = np.zeros_like(x_v)
xddot_v = np.zeros_like(x_v)

U2modes_interp = CubicSpline(x, U2_modes, axis=0)

# y_b and y_b_dot: modal coordinates
y_b = np.zeros((n_steps, n_modes_b))
y_b_dot = np.zeros((n_steps, n_modes_b))

I = np.eye(n_modes_b)

force_contact = np.empty((num_axles, len(time)))
force_contact[:] = np.nan
force_contact[0,0] = -(ms[1] + m_carriage/num_axles) * g

U2 = np.zeros((U2_modes.shape[0], len(time)))
U2_contacts = np.zeros((num_axles, len(time)))

f0_spatial = 2*wn_v[0]/(2*np.pi)/vel
r_interp = generate_harmonic_profile(np.max(x), len(x), f0_spatial, 0)
dr_dx_interp = r_interp.derivative()

alphaR, betaR = modal.get_rayleigh_pars(circ_freqs, damping_ratios)
betaR = 0

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

def solve_system(U0: np.ndarray,
                 dotU0: np.ndarray,
                 ddotU0: np.ndarray,
                 r_c: np.ndarray,
                 dr_dx_c: np.ndarray,
                 dof_inside,
                 fade_weights: np.ndarray):

    yb0 = U0[:n_modes_b]
    xv0 = U0[n_modes_b:]

    dotyb0 = dotU0[:n_modes_b]
    dotxv0 = dotU0[n_modes_b:]

    ddotyb0 = ddotU0[:n_modes_b]
    ddotxv0 = ddotU0[n_modes_b:]

    K_extbridge = np.zeros((n_modes_b, n_modes_b))
    for i in range(len(dof_inside)):
        K_extbridge += ks_contact[i] * U2modes_contact[i].reshape(-1,1) @ U2modes_contact[i].reshape(1,-1)

    C_extbridge = np.zeros((n_modes_b, n_modes_b))
    for i in range(len(dof_inside)):
        C_extbridge += cs_contact[i] * U2modes_contact[i].reshape(-1,1) @ U2modes_contact[i].reshape(1,-1)

    Kb = np.diag(circ_freqs**2) + K_extbridge
    Mb = np.eye(n_modes_b)
    Cb = alphaR*np.eye(n_modes_b) + betaR*np.diag(circ_freqs**2)
    Kbb = Kb + 2/dt*(Cb + C_extbridge) + 4/dt**2*Mb

    Kbv = np.zeros((n_modes_b, n_modes_v))
    # broadcasting
    Kbv[:,dof_inside] += -(U2modes_contact * ks_contact.reshape(-1,1)).T
    Kbv[:,dof_inside] += -2/dt*(U2modes_contact * cs_contact.reshape(-1,1)).T

    Kvb = np.zeros((n_modes_v, n_modes_b))
    Kvb[dof_inside,:] += -ks_contact.reshape(-1,1) * U2modes_contact
    Kvb[dof_inside,:] += -2/dt * cs_contact.reshape(-1,1) * U2modes_contact
    Kvv =  Kv + 2/dt*Cv + 4/dt**2*Mv

    Keff = np.block([
            [Kbb, Kbv],
            [Kvb, Kvv]
    ])

    mass_per_axle = ms[1+idx_inside] + m_carriage/num_axles

    fb = np.zeros(n_modes_b)

    # === Apply fade ONLY to gravity ===
    gravity_term = -(mass_per_axle * g * fade_weights).reshape(-1, 1)
    fb += np.sum(U2modes_contact * gravity_term, axis=0)

    # === No fade on spring/damper terms ===
    spring_damper_term = (ks_contact * r_c + cs_contact * vel * dr_dx_c).reshape(-1,1)
    fb += -np.sum(U2modes_contact * spring_damper_term, axis=0)

    fbM = (4/dt**2*yb0 + 4/dt * dotyb0 + ddotyb0)
    fbM = fbM.reshape(-1,1)
    fb += (Mb @ fbM).squeeze()

    fbC = 2/dt*yb0 + dotyb0 + dt/2 * ddotyb0
    fbC = fbC.reshape(-1,1)
    fb += (Cb @ fbC).squeeze()
    # contribution to feff related to dashpot - bridge on bridge
    fb += (C_extbridge @
            (2/dt*yb0 + dotyb0 + dt/2 * ddotyb0).reshape(-1,1)).squeeze()
    # contribution to feff related to dashpot - vehicle on bridge
    fb += -((U2modes_contact * cs_contact.reshape(-1,1)).T @ (
            2/dt * xv0[dof_inside] + dotxv0[dof_inside] + dt/2 * ddotxv0[dof_inside]).reshape(-1,1)).squeeze()

    fv = np.zeros(n_modes_v)
    fv[idx_inside] += ks_contact * r_c + cs_contact * vel * dr_dx_c
    fvM = (4/dt**2*xv0 + 4/dt * dotxv0 + ddotxv0)
    fvM = fvM.reshape(-1,1)
    fv += (Mv @ fvM).squeeze()

    fvC = 2/dt*xv0 + dotxv0 + dt/2 * ddotxv0
    fvC = fvC.reshape(-1,1)
    fv += (Cv @ fvC).squeeze()
    # contribution to feff related to dashpot - bridge on vehicle
    fv[dof_inside] += -((U2modes_contact * cs_contact.reshape(-1,1)) @ (
            2/dt*yb0 + dotyb0 + dt/2 * ddotyb0).reshape(-1,1)).squeeze()

    feff = np.concatenate((fb, fv))

    U = np.linalg.solve(Keff, feff)

    ddotU = 4/dt**2*(U - U0) - 4/dt*dotU0 - ddotU0
    dotU = dotU0 + dt/2*(ddotU0 + ddotU)

    return U, dotU, ddotU

def get_fade_weights(ramp_duration, apply_fade: bool) -> np.ndarray:
    fade_weights = np.ones_like(ks_contact)

    if apply_fade:
        for j, axle_idx in enumerate(idx_inside):
            t_entry = (0 - xc0[axle_idx]) / vel
            t_exit  = (length_b - xc0[axle_idx]) / vel

            t_rel_entry = t - t_entry
            t_rel_exit = t_exit - t

            # Fade in
            if 0 <= t_rel_entry <= ramp_duration:
                fade_weights[j] = 0.5 * (1 - np.cos(np.pi * t_rel_entry / ramp_duration))

            # Fade out
            elif 0 <= t_rel_exit <= ramp_duration:
                fade_weights[j] = 0.5 * (1 - np.cos(np.pi * t_rel_exit / ramp_duration))

    return fade_weights

U = np.zeros(n_modes_b + n_modes_v)
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
    apply_fade = True
    fade_weights = get_fade_weights(ramp_duration, apply_fade)

    r_c = r_interp(x_c)
    dr_dx_c = dr_dx_interp(x_c)

    U0 = U
    dotU0 = dotU
    ddotU0 = ddotU

    U2modes_contact = U2modes_interp(x_c).reshape(len(dof_inside),-1)

    U, dotU, ddotU = solve_system(U0, dotU0, ddotU0, r_c, dr_dx_c, idx_inside, fade_weights)

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

idx_axle_plot = 1
logging.info('ERRORS')
plt.figure()
y = force_contact[idx_axle_plot]
plt.plot(time, y)
plt.xlabel(r'$t$')
plt.ylabel(r'$F_c$')
plt.title('Contact force')
plt.show()

dofv = 3
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
plt.figure()
plt.plot(time, U2_mid, label='interaction')
plt.plot(time, U2_noint_mid, label='no interaction')
plt.xlabel(r'$t$')
plt.ylabel(r'$U_2$')
plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.title('Midspan displacement')
plt.legend()
plt.show()

## Verification
dict_matlab = loadmat('VerificationVBI.mat')
U2_mat = dict_matlab['U_xt'].squeeze()
t_mat = dict_matlab['t_ver'].squeeze()
F_contact_mat = dict_matlab['F_contact'].squeeze()
midnode = dict_matlab['node_midspan'].item()
U2mid_mat = U2_mat[midnode]
x_v_mat = dict_matlab['U_veh'].squeeze()

# Midpoint displacement
U2mid_mat = interpolate_mat(t_mat, U2mid_mat, time)
err = get_err(U2mid_mat, U2_mid)
logging.info(f'ERROR MATLAB - U2 MIDSPAN = {err}')

plt.figure()
plt.plot(time, U2mid_mat, label='Matlab')
plt.plot(time, U2_mid, label='Python')
#plt.plot(time, U2_mid, label='Python - BKEUL')
plt.xlabel(r'$t$')
plt.ylabel(r'$U_2$')
plt.legend()
plt.title('Midspan displacement verification')
plt.show()

x_v_mat = interpolate_mat(t_mat, x_v_mat, time)
err = get_err(x_v, x_v_mat)
logging.info(f'ERROR MATLAB - VEHICLE = {err}')
plt.figure()
if len(x_v_mat.shape) != 1:
    x_v_mat = x_v_mat[dofv]
plt.plot(time, x_v_mat, label='Matlab')
plt.plot(time, x_v[dofv], label='Python')
plt.xlabel(r'$t$')
plt.ylabel(r'$x_v$')
plt.title('Vehicle displacement verification')
plt.legend()
plt.show()

# Contact force
F_contact_mat = interpolate_mat(t_mat, F_contact_mat[idx_axle_plot], time)
err = get_err(F_contact_mat, force_contact[idx_axle_plot])
logging.info(f'ERROR MATLAB - CONTACT FORCE = {err}')

plt.figure()
plt.plot(time, F_contact_mat, label='Matlab')
plt.plot(time, force_contact[idx_axle_plot], label='Python')
plt.xlabel(r'$t$')
plt.ylabel(r'$F$')
plt.title('Contact force verification')
plt.legend()
plt.show()
