import torch
import numpy as np

kB_eV_K = 8.617333262145e-5
eV_A_to_N = 1.602176634e-9
ms2_to_Afs2 = 1e-20
amu_to_kg = 1.66053906660e-27


class LangevinBAOBA:
    def __init__(
        self,
        atoms,
        dt_fs=0.5,
        temperature_K=300.0,
        gamma_fs=0.01,
        device="cuda",
        dtype=torch.float64,
        seed=None,
    ):
        self.atoms = atoms
        self.dt = torch.tensor(dt_fs, device=device, dtype=dtype)
        self.gamma = torch.tensor(gamma_fs, device=device, dtype=dtype)
        self.T = temperature_K
        self.device = device
        self.dtype = dtype

        if seed is not None:
            torch.manual_seed(seed)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)

        # --- state ---
        self.x = torch.tensor(atoms.get_positions(), device=device, dtype=dtype)  # Å
        self.v = self._init_velocities()  # Å/fs

        self.m = (
            torch.tensor(atoms.get_masses(), dtype=dtype)
            * amu_to_kg
        ).to(device)

        self.inv_m = 1.0 / self.m

    # -----------------------------------------------------

    def _init_velocities(self):
        m = torch.tensor(self.atoms.get_masses()) * amu_to_kg
        kT_J = kB_eV_K * self.T * 1.602176634e-19
        std = torch.sqrt(kT_J / m)  # m/s
        v = torch.randn((len(m), 3)) * std[:, None]
        return (v * 1e-5).to(self.device, self.dtype)  # Å/fs

    # -----------------------------------------------------

    def _forces(self):
        """Forces from atoms.calc (eV/Å)"""
        self.atoms.set_positions(self.x.cpu().numpy())
        return torch.tensor(self.atoms.get_forces(), device=self.device, dtype=self.dtype)

    # -----------------------------------------------------

    def _acceleration(self, f):
        f_N = f * eV_A_to_N
        a_ms2 = f_N * self.inv_m[:, None]
        return a_ms2 * ms2_to_Afs2  # Å/fs²

    # -----------------------------------------------------

    def step(self, nsteps=1): # BAOBA
        dt = self.dt
        gamma = self.gamma

        kT_J = (
            kB_eV_K * self.T * 1.602176634e-19
        )
        c = torch.exp(-gamma * dt)
        sigma = torch.sqrt(
            (1 - c**2) * (kT_J / self.m) * 1e-10
        )

        for _ in range(nsteps):
            # B
            a = self._acceleration(self._forces())
            self.v += 0.5 * dt * a

            # A
            self.x += 0.5 * dt * self.v

            # O
            R = torch.randn_like(self.v)
            self.v = c * self.v + sigma[:, None] * R

            # B
            a = self._acceleration(self._forces())
            self.v += 0.5 * dt * a

            # A
            self.x += 0.5 * dt * self.v
            
        self.atoms.set_velocities(self.v.cpu().numpy())

    # -----------------------------------------------------

    def temperature(self):
        v_ms = self.v * 1e5
        KE = 0.5 * torch.sum(self.m[:, None] * v_ms**2)
        dof = 3 * len(self.m)
        return (2 * KE / (dof * 1.380649e-23)).item()
