import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt

from load_els import LoadElement, LoadSystem

class Beam:
    def __init__(self, length, mu, E, J, xi, n_modes, nx, nt, c):
        self.length = length
        self.mu = mu
        self.E = E
        self.J = J
        self.n_modes = n_modes

        self.x = np.linspace(0, length, nx)
        self.nt = nt
        self.t = np.linspace(0, length/c, self.nt)
        self.c = c
        self.omega = self.c * np.pi/length
        self.alpha = self.omega/self.return_omega_j(1)
        self.omega_d = self.return_omega_j(1) * xi

        self.g = 9.81

    def get_v0(self, P):
        v0 = 2*P*self.length**3/(np.pi**4*self.E*self.J)
        return v0

    def get_M0(self, P):
        M0 = P * self.length/4
        return M0

    def return_omega_j(self, j):
        omega_j = j ** 2 * np.pi ** 2 / self.length ** 2 * (self.E * self.J / self.mu) ** (1 / 2)
        return omega_j

    def get_v(self, alpha, v0, return_contr: bool = 0):
        gridx, gridt = np.meshgrid(self.x, self.t, indexing='ij')

        v = np.zeros_like(gridx)
        beta = self.omega_d / self.return_omega_j(1)

        v_contr = []
        for j in range(1, self.n_modes + 1):
            omega_j = self.return_omega_j(j)
            v_j = np.sin(j * np.pi * gridx/self.length)
            if j == alpha:
                if beta != 0:
                    v_j *= (np.exp(-self.omega_d*gridt)*np.sin(
                        j*self.omega*gridt)- j**2/beta * np.cos(j*self.omega*gridt) * (1-
                                                                             np.exp(-self.omega_d*gridt)))
                else:
                    v_j *= (np.sin(j * self.omega * gridt) -
                            j * self.omega * gridt * np.cos(j * self.omega * gridt))
                v_j *= 1/(2*j**4)
            else:
                v_j *= (np.sin(self.omega * j * gridt) -
                        alpha/j * np.exp(-self.omega_d * gridt) * np.sin(omega_j * gridt))
                v_j *= 1/(j**2*(j**2-alpha**2))

            v += v_j
            v_contr.append(np.mean(v_j[v_j.shape[0]//2, :]))

        v *= v0
        v_contr = np.array(v_contr)

        return v


    def get_v_dot(self, alpha, t, v0: float):
        gridx, gridt = np.meshgrid(self.x, t, indexing='ij')

        v = np.zeros_like(gridx)
        beta = self.omega_d / self.return_omega_j(1)

        for j in range(1, self.n_modes + 1):
            omega_j = self.return_omega_j(j)
            v_j = np.sin(j * np.pi * gridx/self.length)
            if j == alpha:
                if beta != 0:
                    pass
                else:
                    v_j *= (j*self.omega*np.cos(j * self.omega * gridt) -
                            (j * self.omega * np.cos(j * self.omega * gridt) -
                             j**2*self.omega**2*gridt*np.sin(j * self.omega * gridt)))
                v_j *= 1/(2*j**4)
            else:
                v_j *= (j*self.omega*np.cos(self.omega * j * gridt) -
                        alpha/j * (-self.omega_d *np.exp(-self.omega_d * gridt) * np.sin(omega_j * gridt) +
                                   np.exp(-self.omega_d * gridt) * omega_j * np.cos(omega_j * gridt)))
                v_j *= 1/(j**2*(j**2-alpha**2))

            v += v_j

        v *= v0

        return v


    def get_bm(self, alpha, M0):
        gridx, gridt = np.meshgrid(self.x, self.t, indexing='ij')

        bm = np.zeros_like(gridx)
        beta = self.omega_d / self.return_omega_j(1)

        for j in range(1, self.n_modes+1):
            omega_j = self.return_omega_j(j)
            bm_j = np.sin(j * np.pi * gridx/self.length)
            if alpha == j:
                if beta != 0:
                    bm_j *= 1/(np.pi**2*j**2)*(np.exp(-self.omega_d*gridt)*np.sin(
                        j*self.omega*gridt)-j**2/beta*np.cos(j*self.omega*gridt)*(1-
                                                                        np.exp(-self.omega_d*gridt)))
                else:
                    bm_j *= 1/(np.pi**2*j**2)*(np.sin(j*self.omega*gridt)-
                                              j*self.omega*gridt*np.cos(j*self.omega*gridt))
                bm_j *= 4*M0
            else:
                bm_j *= (np.sin(self.omega * j * gridt)
                        - alpha/j * np.exp(-self.omega_d * gridt) * np.sin(omega_j * gridt))
                bm_j *= 1/(j**2*(1-alpha**2/j**2))
                bm_j *= 8/(np.pi**2)
                bm_j *= M0

            bm += bm_j

        return bm

    def build_global_time(self, loadsystem : LoadSystem) -> np.ndarray:
        last_location = loadsystem.elements[-1].location[-1]
        global_t = np.linspace(0, (self.length + np.abs(last_location))/self.c, self.nt)

        return global_t


    def compute_multi_response(self, loadsystem : LoadSystem) -> Tuple:
        t_global = self.build_global_time(loadsystem)
        total_disp = np.zeros((len(self.x), len(t_global)))
        total_bm = np.zeros_like(total_disp)

        for el in loadsystem.elements:
            for i, m_axle in enumerate(el.m_per_axle):
                t_enter = np.abs(el.location[i])/self.c

                v0 = self.get_v0(m_axle * self.g)
                M0 = self.get_M0(m_axle * self.g)

                vi_forced = self.get_v(self.alpha, v0)
                bm_forced = self.get_bm(self.alpha, M0)
                vdot_last = self.get_v_dot(self.alpha, v0, self.t[-1])
                t_free = np.linspace(0, t_global[-1] - self.t[-1] - t_enter, self.nt)
                vi_free, bm_free = self.compute_free_response(vi_forced[:,-1], vdot_last, t_free)

                ti = np.concatenate((self.t, self.t[-1] + t_free))
                vi = np.concatenate((vi_forced, vi_free), axis=1)
                bmi = np.concatenate((bm_forced, bm_free), axis=1)

                vi_interp = interp1d(ti, vi, axis=1, bounds_error=False, fill_value=0.0)
                bmi_interp = interp1d(ti, bmi, axis=1, bounds_error=False, fill_value=0.0)

                vi_in_global = vi_interp(t_global - t_enter)
                bmi_in_global = bmi_interp(t_global - t_enter)

                total_disp += vi_in_global
                total_bm += bmi_in_global

        return t_global, total_disp, total_bm


    def get_modes_shapes(self):
        phi = np.zeros((len(self.x), self.n_modes))  # Initialize matrix

        for j in range(1, self.n_modes + 1):
            phi[:, j - 1] = np.sin(j * np.pi * self.x / self.length)

        return phi


    def compute_free_response(self, v0: np.ndarray, v0_dot: np.ndarray, t_free):
        phi = self.get_modes_shapes()

        """
        plt.figure()
        plt.plot(self.x, v0)
        plt.plot(self.x, v0_dot)
        plt.show()
        """

        a = simpson(v0.reshape(-1,1) * phi, self.x, axis=0)
        b = simpson(v0_dot.reshape(-1,1) * phi, self.x, axis=0)

        a *= 2 / self.length
        b *= 2 / self.length

        for j in range(1, b.shape[0] + 1):
            b[j - 1] *= 1 / self.return_omega_j(j)

        gridx, gridt = np.meshgrid(self.x, t_free, indexing='ij')

        v = np.zeros((len(self.x), len(t_free)))
        bm = np.zeros((len(self.x), len(t_free)))

        alpha = 36
        p = 4

        for j in range(0, a.shape[0]):
            omega_j = self.return_omega_j(j + 1)
            vj = a[j] * np.cos(omega_j * gridt) + b[j] * np.sin(omega_j * gridt)
            vj *= np.sin((j + 1) * np.pi * gridx / self.length)
            v += vj

            bmj = a[j] * np.cos(omega_j * gridt) + b[j] * np.sin(omega_j * gridt)
            bmj *= np.sin((j + 1) * np.pi * gridx / self.length)
            bmj *= (j+1)**2 * np.pi**2 / self.length**2
            # Filter due to modal truncation, v0_dot -> b[j] != 0
            # amplification for the high modes
            sigma_j = np.exp(-alpha * ((j+1)/a.shape[0])**p)
            bmj *= sigma_j
            bmj *= self.E*self.J
            bm += bmj

        return v, bm
