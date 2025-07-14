import numpy as np
import pandas as pd

def get_modes(E: float, J: float, mu: float, x: np.ndarray, n_modes_b: int, from_FEM: bool):
    L = np.max(x)
    i = np.arange(1, n_modes_b+1)

    if from_FEM:
        freqs = np.loadtxt('freqs.csv', delimiter=',', skiprows=1)
        circ_freqs = 2*np.pi*freqs
        U1_modes = pd.read_csv('U1.csv', delimiter=',').iloc[:,1:].to_numpy()
        U2_modes = pd.read_csv('U2.csv', delimiter=',').iloc[:,1:].to_numpy()
        freqs_an = (i*np.pi/L)**2*np.sqrt(E*J/mu)/(2*np.pi)
        np.savetxt("freqs_an.csv", freqs_an, delimiter=",")
    else:
        circ_freqs = (i*np.pi/L)**2*np.sqrt(E*J/mu)
        freqs = circ_freqs/(2*np.pi)
        U2_modes = np.sin(np.pi*i.reshape(1,-1)*x.reshape(-1,1)/L)
        mass_norm = np.sqrt(mu*L/2)
        U2_modes /= mass_norm
        np.savetxt("freqs_an.csv", freqs, delimiter=",")
        np.savetxt("U2_modes_an.csv", U2_modes, delimiter=",")

    return U2_modes, freqs


def get_rayleigh_pars(circ_freqs: np.ndarray, damping_ratios: list):
    if len(damping_ratios) != 2:
        raise ValueError(f"Expected 2 damping ratios, got {len(damping_ratios)}.")

    A = np.array([[1/circ_freqs[0], circ_freqs[0]], [1/circ_freqs[1], circ_freqs[1]]])
    A *= 1/2
    b = np.array(damping_ratios)
    raypars = np.linalg.solve(A, b)
    alphaR = raypars[0]
    betaR = raypars[1]

    return alphaR, betaR
