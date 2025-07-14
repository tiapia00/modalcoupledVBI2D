import numpy as np


def get_fade_weights(ramp_duration, apply_fade: bool, idx_inside: np.ndarray, xc0: np.ndarray, vel: float, t: float, length_b: float) -> np.ndarray:
    fade_weights = np.ones_like(idx_inside)

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


def solve_system(U0: np.ndarray,
                 dotU0: np.ndarray,
                 ddotU0: np.ndarray,
                 config: dict):


    # Unpacking dict
    n_modes_b = config["n_modes_b"]
    n_modes_v = config["n_modes_v"]

    U2modes_contact = config["U2modes_contact"]
    ks_contact = config["ks_contact"]
    cs_contact = config["cs_contact"]

    Kv = config["Kv"]
    Cv = config["Cv"]
    Mv = config["Mv"]

    wn_b = config["wn_b"]

    alphaR = config["alphaR"]
    betaR = config["betaR"]

    ms = config["ms"]
    m_carriage = config["m_carriage"]
    num_axles = config["num_axles"]

    idx_inside = config["idx_inside"]

    vel = config["vel"]

    g = config["g"]
    dt = config["dt"]

    r_c = config["r_c"]
    dr_dx_c = config["dr_dx_c"]

    dof_inside = config["dof_inside"]
    fade_weights = config["fade_weights"]


    # Solver
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

    Kb = np.diag(wn_b**2) + K_extbridge
    Mb = np.eye(n_modes_b)
    Cb = alphaR*np.eye(n_modes_b) + betaR*np.diag(wn_b**2)
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
