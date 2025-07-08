import numpy as np
import matplotlib.pyplot as plt

def quarter_car(ms: list, cs: np.ndarray, ks: np.ndarray):
    """
    Quarter Car Model with multiple axles

    ms = n_axles + 1
    ks, cs = 2 x n_axles
    DOFs ordered clockwise starting from vehicle mass
    No pitch

    ↑ Positive direction

        Single Vehicle Mass (m1)
       ┌───────────────────────────┐
       │                           │
       └─────┬─────────┬───────────┘
             │         │
         ┌───┴───┐ ┌───┴───┐     ...   ┌───┴───┐
         │       │ │       │           │       │
        [c1₁]  [k1₁]     [c1₂]  [k1₂] ...     [c1ₙ]  [k1ₙ]
         │       │ │       │           │       │
       ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ...   ┌─┴─┐   ┌─┴─┐
       │m2₁│   │m2₂│   │m2₃│         │m2ₙ│
       └───┘   └───┘   └───┘         └───┘
         │       │     │               │
        [c2₁]  [k2₁] [c2₂]  [k2₂] ... [c2ₙ]  [k2ₙ] <- accounted as external forces, so not present
         │       │     │               │
     ───────────────────────────────────────────
                   Ground / Deck
    """

    M_v = np.diag(ms)
    n = M_v.shape[0] - 1

    K_v = np.zeros_like(M_v)
    K_v[0,0] = np.sum(ks[0])
    for i in range(n):
        K_v[0, i+1] = -ks[0,i]
        K_v[i+1, 0] = -ks[0,i]
        K_v[i+1, i+1] = ks[0,i] + ks[1,i]

    C_v = np.zeros_like(M_v)
    C_v[0,0] = np.sum(cs[0])
    for i in range(n):
        C_v[0, i+1] = -cs[0,i]
        C_v[i+1, 0] = -cs[0,i]
        C_v[i+1, i+1] = cs[0,i] + cs[1,i]

    return (M_v, C_v, K_v)

def quarter_car_pitch(ms: np.ndarray, I: list, cs: np.ndarray, ks: np.ndarray, l_veh: float):
    """
    Quarter Car Model with multiple axles

    ms = n_axles + 1
    ks, cs = 2 x n_axles
    ks[0,0] -> front stiffness
    DOFs ordered clockwise starting from vehicle mass
    theta: first entry
    positive rotation if counterclockwise

    ↑ Positive direction

        Single Vehicle Mass (m1)
       ┌───────────────────────────┐
       │                           │
       └─────┬─────────┬───────────┘
             │         │
         ┌───┴───┐ ┌───┴───┐     ...   ┌───┴───┐
         │       │ │       │           │       │
        [c0n]  [k0n]     [c02]  [k02] ...     [c00]  [k00]
         │       │ │       │           │       │
       ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ...   ┌─┴─┐   ┌─┴─┐
       │m2₁│   │m2₂│   │m2₃│         │m2ₙ│
       └───┘   └───┘   └───┘         └───┘
         │       │     │               │
        [c1n]  [k1n] [c12]  [k12] ... [c10]  [k10] <- accounted as external forces, so not present
         │       │     │               │
     ───────────────────────────────────────────
                   Ground / Deck
    """

    M_v = np.diag(np.concatenate((ms, I)))
    n = ms.shape[0] - 1

    K_v = np.zeros_like(M_v)
    K_v[0,0] = np.sum(ks[0])
    for i in range(n):
        K_v[0, i+1] = -ks[0,i]
        K_v[i+1, 0] = -ks[0,i]
        K_v[i+1, i+1] = ks[0,i] + ks[1,i]

    C_v = np.zeros_like(M_v)
    C_v[0,0] = np.sum(cs[0])
    for i in range(n):
        C_v[0, i+1] = -cs[0,i]
        C_v[i+1, 0] = -cs[0,i]
        C_v[i+1, i+1] = cs[0,i] + cs[1,i]

    # contributions due to pitch of the vehicle, linearized wrt to theta=0
    K_v[0,-1] = ks[0,0]*l_veh/2 - ks[0,1]*l_veh/2
    K_v[1,-1] = -ks[0,0]*l_veh/2
    K_v[2,-1] = ks[0,1] * l_veh/2
    K_v[-1, 0] = ks[0,0]*l_veh/2 - ks[0,1]*l_veh/2
    K_v[-1, 1] = -ks[0,0]*l_veh/2
    K_v[-1, 2] = ks[0,1]*l_veh/2
    K_v[-1, -1] = (ks[0,0] + ks[0,1]) * (l_veh**2) / 4

    C_v[0,-1] = cs[0,0]*l_veh/2 - cs[0,1]*l_veh/2
    C_v[1,-1] = -cs[0,0]*l_veh/2
    C_v[2,-1] = cs[0,1] * l_veh/2
    C_v[-1, 0] = cs[0,0]*l_veh/2 - cs[0,1]*l_veh/2
    C_v[-1, 1] = -cs[0,0]*l_veh/2
    C_v[-1, 2] = cs[0,1]*l_veh/2
    C_v[-1, -1] = (cs[0,0] + cs[0,1]) * (l_veh**2) / 4

    n = M_v.shape[0]
    idx = np.concatenate(([n-1], np.arange(n-1)))

    M_v = M_v[np.ix_(idx, idx)]
    C_v = C_v[np.ix_(idx, idx)]
    K_v= K_v[np.ix_(idx, idx)]

    return (M_v, C_v, K_v)


def plot_modes(modes: np.ndarray, wn_v: np.ndarray, xv_eq: np.ndarray, l_veh: float):
    plt.figure()
    h_plot = 5
    theta_eq = xv_eq[0]
    z_eq = [0, xv_eq[1] + h_plot]
    z_f_eq = [l_veh/2, xv_eq[2] + h_plot/2]
    z_r_eq = [-l_veh/2, xv_eq[3] + h_plot/2]

    n_modes_v = modes.shape[0]

    def plot_rotation_line_at_eq(z_eq_point, angle, length=0.5, **kwargs):
        x_eq, y_eq = z_eq_point
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        plt.plot([x_eq, x_eq + dx], [y_eq, y_eq + dy], **kwargs)

    for i in range(n_modes_v):
        mode = modes[:,i]
        mode = mode / np.max(np.abs(mode))

        theta = theta_eq.copy()
        theta += mode[0]
        z = z_eq.copy()
        z[1] += mode[1]
        z_f = z_f_eq.copy()
        z_f[1] += mode[2]
        z_r = z_r_eq.copy()
        z_r[1] += mode[3]

        plt.subplot(n_modes_v//2, n_modes_v//2, i+1)
        plot_rotation_line_at_eq(z_eq, theta)
        plt.scatter(*z_eq)
        plt.scatter(*z_f_eq)
        plt.scatter(*z_r_eq)

        plt.scatter(*z, marker='^')
        plt.scatter(*z_f, marker='^')
        plt.scatter(*z_r, marker='^')
        plt.title(rf'$\omega_{i+1} = {wn_v[i]:.2f}$')

    plt.tight_layout()
    plt.show()
