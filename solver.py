import numpy as np
from itertools import combinations


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


class NewmarkSolver:
    def __init__(self,
                 alpha: float,
                 delta: float,
                 dt: float,
                 vehicle_matrs: tuple,
                 n_axles: int,
                 bridge_matrs: tuple,
                 dof_contact_start: int,
                 x0: np.ndarray,
                 mass_per_axle: np.ndarray,
                 stiff_contact: float):

        self.alpha = alpha
        self.delta = delta

        self.dt = dt
        self.a0 = 1/(self.alpha*self.dt**2)
        self.a1 = self.delta/(self.alpha*self.dt)
        self.a2 = 1/(self.alpha*self.dt)
        self.a3 = 1/(2*self.alpha) - 1
        self.a4 = self.delta/self.alpha - 1
        self.a5 = (self.dt*self.delta)/(2*self.alpha) - self.dt

        self.Kv, self.Cv, self.Mv = vehicle_matrs
        self.n_axles = n_axles
        self.Kb, self.Cb, self.Mb = bridge_matrs
        self.dof_contact_start = dof_contact_start

        self.x0 = x0
        self.mass_per_axle = mass_per_axle
        self.stiff_contact = stiff_contact


    def solve_system(self,
                     U0: np.ndarray,
                     dotU0: np.ndarray,
                     ddotU0: np.ndarray,
                     n_modes_b: int,
                     U2modes_contact: np.ndarray,
                     rough: np.ndarray,
                     idx_outside: np.ndarray):


        # Solver
        yb0 = U0[:n_modes_b]
        xv0 = U0[n_modes_b:]

        dotyb0 = dotU0[:n_modes_b]
        dotxv0 = dotU0[n_modes_b:]

        ddotyb0 = ddotU0[:n_modes_b]
        ddotxv0 = ddotU0[n_modes_b:]

        # wheel location matrix
        I_v = np.zeros((self.Kv.shape[0], self.n_axles))
        I_v[I_v.shape[0] - self.n_axles:, :self.n_axles] = np.eye(self.n_axles)

        Keff_v = self.Kv + self.a0*self.Mv + self.a1*self.Cv

        inv_Keff_v, alpha_1 = self.get_alpha(Keff_v, I_v)
        beta_1 = self.get_beta_vehicle(inv_Keff_v, xv0, dotxv0, ddotxv0)

        Keff_b = self.Kb + self.a0 * self.Mb + self.a1 * self.Cb

        inv_Keff_b, alpha_2 = self.get_alpha(Keff_b, U2modes_contact.T)
        beta_2 = self.get_beta_bridge(inv_Keff_b, yb0, dotyb0, ddotyb0)

        alpha = np.concatenate((alpha_2, alpha_1))
        beta = np.concatenate((beta_2, beta_1))

        A = np.concatenate((U2modes_contact.T, -I_v), axis=0)

        A_LCP = np.eye(self.n_axles) + A.T @ alpha
        is_p = is_p_matrix(A_LCP)
        if not is_p:
            raise RuntimeError("M is not a P-matrix: LCP solution not unique")

        B_LCP = A.T @ beta - rough - self.x0

        # lambda: compression variable
        lambdas = self.projected_gauss_seidel(A_LCP, B_LCP)

        # At equilibrium, reaction force must be 0 outside, otherwise I would have a response
        lambdas[idx_outside] = 0

        forces_contact = self.stiff_contact * - lambdas

        U = alpha @ forces_contact + beta
        weight_v_contr = - 9.81 * U2modes_contact.T @ self.mass_per_axle
        U[:n_modes_b] += inv_Keff_b @ weight_v_contr

        ddotU = self.update_acc(U, U0, dotU0, ddotU0)
        dotU = self.update_speed(ddotU, dotU0, ddotU0)

        return U, dotU, ddotU, forces_contact


    def get_alpha(self, Keff: np.ndarray, right_term: np.ndarray) -> tuple:
        inv_Keff = np.linalg.inv(Keff)
        alpha = inv_Keff @ right_term
        return inv_Keff, alpha


    def get_beta_vehicle(self,
                 inv_Keff_v: np.ndarray,
                 displ: np.ndarray,
                 vel: np.ndarray,
                 acc: np.ndarray) -> np.ndarray:

        beta = inv_Keff_v @ (self.Mv @ (self.a0 * displ + self.a2 * vel + self.a3 * acc) +
                self.Cv @ (self.a1 * displ + self.a4 * vel + self.a5 * acc))

        return beta

    def get_beta_bridge(self,
                 inv_Keff_b: np.ndarray,
                 displ: np.ndarray,
                 vel: np.ndarray,
                 acc: np.ndarray) -> np.ndarray:

        beta = inv_Keff_b @ (self.Mb @ (self.a0 * displ + self.a2 * vel + self.a3 * acc) +
                self.Cb @ (self.a1 * displ + self.a4 * vel + self.a5 * acc))

        return beta


    @staticmethod
    def projected_gauss_seidel(M, q, x0=None, max_iter=1000, tol=1e-8):
        """
        Solves the LCP: x >= 0, Mx + q >= 0, x^T (Mx + q) = 0
        using Projected Gauss-Seidel iteration.

        Parameters
        ----------
        M : (n, n) array_like
            Positive definite or positive semidefinite matrix.
        q : (n,) array_like
            Vector in the LCP.
        x0 : (n,) array_like, optional
            Initial guess. Defaults to zeros.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        x : (n,) ndarray
            Solution to the LCP.
        """

        M = np.array(M, dtype=float)
        q = np.array(q, dtype=float)
        n = len(q)

        if x0 is None:
            x = np.zeros(n)
        else:
            x = np.array(x0, dtype=float)

        for _ in range(max_iter):
            x_old = x.copy()
            for i in range(n):
                # Standard Gauss-Seidel update
                sigma = np.dot(M[i, :], x) - M[i, i] * x[i]
                x[i] = max(0.0, (-q[i] - sigma) / M[i, i])  # projection onto positive orthant

            # Check convergence (can use norm or max norm)
            if np.linalg.norm(x - x_old, ord=np.inf) < tol:
                break

        return x


    def update_acc(self, U: np.ndarray, U0: np.ndarray, dotU0: np.ndarray, ddotU0: np.ndarray):
        acc = self.a0*(U - U0) - self.a2*dotU0 - self.a3*ddotU0
        return acc


    def update_speed(self, ddotU: np.ndarray, dotU0: np.ndarray, ddotU0: np.ndarray):
        speed = dotU0 + self.delta*self.dt*ddotU + (1 - self.delta)*self.dt*ddotU0
        return speed


def is_p_matrix(M: np.ndarray, tol=1e-12):

    """
    Check if a square matrix M is a P-matrix
    (all principal minors are strictly positive).

    Parameters
    ----------
    M : (n, n) array_like
        The matrix to test.
    tol : float
        Numerical tolerance for considering positivity.

    Returns
    -------
    bool : True if M is a P-matrix, False otherwise.
    """
    M = np.array(M, dtype=float)
    n = M.shape[0]

    if M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")

    # Loop over all sizes of principal minors (1x1, 2x2, ..., nxn)
    for k in range(1, n + 1):
        for rows in combinations(range(n), k):
            submatrix = M[np.ix_(rows, rows)]
            det_val = np.linalg.det(submatrix)
            if det_val <= tol:  # strictly positive required
                return False
    return True
