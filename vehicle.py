import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class VehicleModel:
    def __init__(self, ms, cs, ks, l_veh, J_rot, include_pitch=False):
        self.ms = np.asarray(ms)
        self.cs = np.asarray(cs)
        self.ks = np.asarray(ks)
        self.n_axles = len(ms) - 1
        self.l_veh = l_veh
        self.J_rot = J_rot
        self.include_pitch = include_pitch

        self.M, self.C, self.K = self.build_matrices()
        self.dof_labels = self.get_dof_labels()
        self.dof_map = {label: i for i, label in enumerate(self.dof_labels)}

        self.eigenvals, self.modes = self._get_modes()


    def build_matrices(self):
        if self.include_pitch:
            return self._build_pitch_model()
        else:
            return self._build_quarter_car()


    def _build_quarter_car(self):
        # your original `quarter_car` logic here

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

        M_v = np.diag(self.ms)
        n = M_v.shape[0] - 1

        K_v = np.zeros_like(M_v)
        K_v[0,0] = np.sum(self.ks[0])
        for i in range(n):
            K_v[0, i+1] = -self.ks[0,i]
            K_v[i+1, 0] = -self.ks[0,i]
            K_v[i+1, i+1] = self.ks[0,i] + self.ks[1,i]

        C_v = np.zeros_like(M_v)
        C_v[0,0] = np.sum(self.cs[0])
        for i in range(n):
            C_v[0, i+1] = -self.cs[0,i]
            C_v[i+1, 0] = -self.cs[0,i]
            C_v[i+1, i+1] = self.cs[0,i] + self.cs[1,i]

        return (M_v, C_v, K_v)


    def _build_pitch_model(self):
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

        M_v = np.diag(np.concatenate((self.ms, self.J_rot)))
        n = self.ms.shape[0] - 1

        K_v = np.zeros_like(M_v)
        K_v[0,0] = np.sum(self.ks[0])
        for i in range(n):
            K_v[0, i+1] = -self.ks[0,i]
            K_v[i+1, 0] = -self.ks[0,i]
            K_v[i+1, i+1] = self.ks[0,i]
            K_v[i+1, i+1] += self.ks[1,i]

        C_v = np.zeros_like(M_v)
        C_v[0,0] = np.sum(self.cs[0])
        for i in range(n):
            C_v[0, i+1] = -self.cs[0,i]
            C_v[i+1, 0] = -self.cs[0,i]
            C_v[i+1, i+1] = self.cs[0,i]
            C_v[i+1, i+1] += self.cs[1,i]

        # contributions due to pitch of the vehicle, linearized wrt to theta=0
        # theta is the last dof
        K_v[0,-1] = self.ks[0,0]*self.l_veh/2 - self.ks[0,1]*self.l_veh/2
        K_v[1,-1] = -self.ks[0,0]*self.l_veh/2
        K_v[2,-1] = self.ks[0,1] * self.l_veh/2
        K_v[-1, 0] = self.ks[0,0]*self.l_veh/2 - self.ks[0,1]*self.l_veh/2
        K_v[-1, 1] = -self.ks[0,0]*self.l_veh/2
        K_v[-1, 2] = self.ks[0,1]*self.l_veh/2
        K_v[-1, -1] = (self.ks[0,0] + self.ks[0,1]) * (self.l_veh**2) / 4

        C_v[0,-1] = self.cs[0,0]*self.l_veh/2 - self.cs[0,1]*self.l_veh/2
        C_v[1,-1] = -self.cs[0,0]*self.l_veh/2
        C_v[2,-1] = self.cs[0,1] * self.l_veh/2
        C_v[-1, 0] = self.cs[0,0]*self.l_veh/2 - self.cs[0,1]*self.l_veh/2
        C_v[-1, 1] = -self.cs[0,0]*self.l_veh/2
        C_v[-1, 2] = self.cs[0,1]*self.l_veh/2
        C_v[-1, -1] = (self.cs[0,0] + self.cs[0,1]) * (self.l_veh**2) / 4

        n = M_v.shape[0]
        idx = np.concatenate(([n-1], np.arange(n-1)))

        M_v = M_v[np.ix_(idx, idx)]
        C_v = C_v[np.ix_(idx, idx)]
        K_v= K_v[np.ix_(idx, idx)]

        return (M_v, C_v, K_v)


    def _get_modes(self):

        try:
            eigenvals, modes = eigh(self.K, self.M)
        except ValueError as e:
            raise RuntimeError(f'Error computing eigenvalues: {e}')

        if np.any(eigenvals < 0):
            raise ValueError('Negative eigenvalue of car: system not stable')

        return (eigenvals, modes)


    def plot_modes(self):
        # Modes of vehicle extracted as bridge was fixed
        # "Bridge fixed" implies only vehicle modes (no bridge interaction)

        f_ext_v = -9.81 * np.insert(self.ms, 0, 0)
        xv_eq = np.linalg.solve(self.K, f_ext_v)

        h_plot = 5
        theta_eq = xv_eq[0]
        z_eq = [0, xv_eq[1] + h_plot]
        z_f_eq = [self.l_veh/2, xv_eq[2] + h_plot/2]
        z_r_eq = [-self.l_veh/2, xv_eq[3] + h_plot/2]

        n_modes = self.modes.shape[0]
        n_cols = n_rows = int(np.ceil(np.sqrt(n_modes)))

        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

        def plot_rotation_line_at_eq(z_eq_point, angle, length=0.5, **kwargs):
            x_eq, y_eq = z_eq_point
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            plt.plot([x_eq, x_eq + dx], [y_eq, y_eq + dy], **kwargs)

        # Keep handles for the global legend
        legend_handles = []
        legend_labels = []

        for i in range(n_modes):
            mode = self.modes[:, i]
            mode = mode / np.max(np.abs(mode))  # normalize mode shape

            theta = theta_eq + mode[0]
            z = [z_eq[0], z_eq[1] + mode[1]]
            z_f = [z_f_eq[0], z_f_eq[1] + mode[2]]
            z_r = [z_r_eq[0], z_r_eq[1] + mode[3]]

            ax = plt.subplot(n_rows, n_cols, i + 1)
            plot_rotation_line_at_eq(z_eq, theta, color='gray')

            if i == 0:
                h1 = ax.scatter(*z_eq, marker='o', color='salmon', label='Static equilibrium')
                ax.scatter(*z_f_eq, marker='o', color='salmon')
                ax.scatter(*z_r_eq, marker='o', color='salmon')

                h2 = ax.scatter(*z, marker='^', color='deepskyblue', label='Modal displacements')
                ax.scatter(*z_f, marker='^', color='deepskyblue')
                ax.scatter(*z_r, marker='^', color='deepskyblue')

                legend_handles.extend([h1, h2])
                legend_labels.extend(['Static equilibrium', 'Modal displacements'])
            else:
                ax.scatter(*z_eq, marker='o', color='salmon')
                ax.scatter(*z_f_eq, marker='o', color='salmon')
                ax.scatter(*z_r_eq, marker='o', color='salmon')

                ax.scatter(*z, marker='^', color='deepskyblue')
                ax.scatter(*z_f, marker='^', color='deepskyblue')
                ax.scatter(*z_r, marker='^', color='deepskyblue')

            ax.set_title(rf'$\omega_{i+1} = {np.sqrt(self.eigenvals[i]):.2f}$')

        # Add one global legend for the figure
        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=2, frameon=False)

        # Adjust layout to leave space for legend
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig("figs/vehicle_modes.png", dpi=300, bbox_inches='tight')
        plt.show()


    def get_dof_labels(self):
        if self.include_pitch:
            labels = ['theta'] + ['z_mass_B'] + [f'z_mass_A{i}' for i in range(self.n_axles)]
        else:
            labels = ['z_mass_B'] + [f'z_mass_A{i}' for i in range(self.n_axles)]
        return labels
