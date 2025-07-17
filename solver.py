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


class NewmarkSolver:
    def __init__(self,
                 alpha: float,
                 delta: float,
                 dt: float,
                 vehicle_matrs: tuple,
                 n_axles: int,
                 bridge_matrs: tuple,
                 ):

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


    def solve_system(self,
                     U0: np.ndarray,
                     dotU0: np.ndarray,
                     ddotU0: np.ndarray,
                     config: dict):


        # Unpacking dict
        n_modes_b = config["n_modes_b"]

        U2modes_contact = config["U2modes_contact"]

        # Solver
        xv0 = U0[:n_modes_b]
        yb0 = U0[n_modes_b:]

        dotxv0 = dotU0[:n_modes_b]
        dotyb0 = dotU0[n_modes_b:]

        ddotxv0 = ddotU0[:n_modes_b]
        ddotyb0 = ddotU0[n_modes_b:]

        # wheel location matrix
        # modify this based on the entrance condition
        I_v = np.zeros((self.Kv.shape[0], self.n_axles))
        I_v[I_v.shape[0] - self.n_axles:, :self.n_axles] = np.eye(self.n_axles)

        Keff_v = self.Kv + self.a0*self.Mv + self.a1*self.Cv

        inv_Keff_v, alpha_1 = self.get_alpha(Keff_v, I_v)
        beta_1 = self.get_beta(inv_Keff_v, xv0, dotxv0, ddotxv0)

        Keff_b = self.Kb + self.a0 * self.Mb + self.a1 * self.Cb

        inv_Keff_b, alpha_2 = self.get_alpha(Keff_b, U2modes_contact)
        beta_2 = self.get_beta(inv_Keff_b, yb0, dotyb0, ddotyb0)

        alpha = np.concatenate((alpha_1, alpha_2))
        beta = np.concatenate((beta_1, beta_2))

        A = np.concatenate((-I_v, U2modes_contact), axis=0)

        """
        reduce the size of this system
        for the DOFs outside: no LCP, pure contact
        for the DOFs inside: LCP

        Easiest thing would be to set modes outisde the bridge = 0 whatever
        and then solve car
        """

        # i'll exclude the dofs outisde from the LCP and keep both at 0 relative displacement
        A_LCP = np.eye(len(dof_inside)) + A.T @ alpha
        #B_LCP = A.T @ beta - self.rough - self.eq_virtual_spring
        B_LCP = A.T @ beta

        lambdas = self.projected_gauss_seidel(A_LCP, B_LCP)

        U = alpha @ lambdas + beta
        ddotU = self.update_acc(U, U0, dotU0, ddotU0)
        dotU = self.update_speed(ddotU, dotU0, ddotU0)

        return U, dotU, ddotU


    def get_alpha(self, Keff: np.ndarray, right_term: np.ndarray) -> tuple:
        inv_Keff = np.linalg.inv(Keff)
        alpha = inv_Keff @ right_term
        return inv_Keff, alpha


    def get_beta(self,
                 inv_Keff: np.ndarray,
                 displ: np.ndarray,
                 vel: np.ndarray,
                 acc: np.ndarray) -> np.ndarray:

        beta = inv_Keff @ (self.Mv @ (self.a0 * displ + self.a2 * vel + self.a3 * acc) +
                self.Cv @ (self.a1 * displ + self.a4 * vel + self.a5 * acc))

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
