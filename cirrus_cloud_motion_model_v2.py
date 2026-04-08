#!/usr/bin/env python3
"""
cirrus_cloud_motion_model_v2.py

A stronger Python 3 model of cirrus cloud motion.

FEATURES
--------
1. Three vertical cirrus layers:
      - lower cirrus layer
      - middle cirrus layer
      - upper cirrus layer

2. Separate horizontal winds in each layer.

3. Vertical exchange between adjacent layers.

4. Temperature-dependent sublimation:
      warmer layers lose cirrus faster.

5. Size-dependent sedimentation:
      larger representative ice particles fall faster.

6. Anisotropic turbulent diffusion.

7. Diagnostics:
      - total mass in each layer
      - domain-integrated total mass
      - center of mass
      - peak location

8. Optional animation.

GOVERNING IDEA
--------------
Each layer k has cloud concentration C_k(x,y,t) satisfying:

  dC_k/dt
    + u_k dC_k/dx + v_k dC_k/dy
    = Kx_k d2C_k/dx2 + Ky_k d2C_k/dy2
      - lambda_sub,k * C_k
      - lambda_fall,k * C_k
      + vertical_exchange
      + source_k

This is not a full microphysical cloud model, but it is stronger than a
single-layer bulk transport model and better suited to cirrus evolution.

DATA DICTIONARY
---------------
nx, ny
    Number of grid points in x and y.

dx, dy [m]
    Grid spacing.

dt [s]
    Time step.

steps
    Number of time steps.

n_layers
    Number of vertical layers, fixed here at 3.

layer_names
    Names of the cirrus layers.

layer_altitudes [m]
    Representative geometric heights of the layers.

layer_depths [m]
    Effective thickness of each layer.

u0[k], v0[k] [m/s]
    Base wind for layer k.

shear_x[k], shear_y[k] [1/s]
    Horizontal wind shear coefficients for layer k.

kx[k], ky[k] [m^2/s]
    Diffusion coefficients for layer k.

temperature[k] [K]
    Representative layer temperature.

particle_radius[k] [m]
    Representative ice particle radius for each layer.

fall_speed[k] [m/s]
    Effective sedimentation speed.

base_sublimation[k] [1/s]
    Baseline sublimation coefficient.

exchange_updown [1/s]
    Vertical exchange coefficient between neighboring layers.

source_strength[k]
    Source amplitude for each layer.

cloud_center_x, cloud_center_y [m]
    Initial cloud center.

cloud_sigma_x[k], cloud_sigma_y[k] [m]
    Initial Gaussian scale lengths per layer.

cloud_peak[k]
    Initial peak concentration for each layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import sys

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required.", file=sys.stderr)
    raise

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass(frozen=True)
class ModelConfig:
    nx: int = 180
    ny: int = 120
    dx: float = 1000.0
    dy: float = 1000.0
    dt: float = 15.0
    steps: int = 960

    n_layers: int = 3

    layer_names: Tuple[str, str, str] = ("Lower cirrus", "Middle cirrus", "Upper cirrus")
    layer_altitudes: Tuple[float, float, float] = (8000.0, 10000.0, 12000.0)
    layer_depths: Tuple[float, float, float] = (1500.0, 1800.0, 2200.0)

    u0: Tuple[float, float, float] = (14.0, 22.0, 31.0)
    v0: Tuple[float, float, float] = (3.0, 5.0, 8.0)

    shear_x: Tuple[float, float, float] = (0.8e-4, 1.4e-4, 2.0e-4)
    shear_y: Tuple[float, float, float] = (-0.3e-4, -0.5e-4, -0.8e-4)

    kx: Tuple[float, float, float] = (80.0, 120.0, 180.0)
    ky: Tuple[float, float, float] = (60.0, 100.0, 140.0)

    temperature: Tuple[float, float, float] = (235.0, 225.0, 215.0)

    particle_radius: Tuple[float, float, float] = (70e-6, 45e-6, 25e-6)
    fall_speed: Tuple[float, float, float] = (0.050, 0.028, 0.012)

    base_sublimation: Tuple[float, float, float] = (3.0e-5, 1.8e-5, 1.0e-5)

    exchange_updown: float = 2.0e-5

    source_strength: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    cloud_center_x: float = 30000.0
    cloud_center_y: float = 35000.0
    cloud_sigma_x: Tuple[float, float, float] = (9000.0, 12000.0, 15000.0)
    cloud_sigma_y: Tuple[float, float, float] = (6000.0, 8500.0, 11000.0)
    cloud_peak: Tuple[float, float, float] = (0.50, 0.85, 1.10)

    save_every: int = 20

    def validate(self) -> None:
        if self.n_layers != 3:
            raise ValueError("This version is implemented for exactly 3 layers.")
        if self.nx < 5 or self.ny < 5:
            raise ValueError("nx and ny must be at least 5.")
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError("dx and dy must be positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.steps < 1:
            raise ValueError("steps must be at least 1.")
        if self.exchange_updown < 0.0:
            raise ValueError("exchange_updown must be non-negative.")
        for arr_name, arr in [
            ("layer_depths", self.layer_depths),
            ("kx", self.kx),
            ("ky", self.ky),
            ("fall_speed", self.fall_speed),
            ("base_sublimation", self.base_sublimation),
            ("cloud_sigma_x", self.cloud_sigma_x),
            ("cloud_sigma_y", self.cloud_sigma_y),
            ("cloud_peak", self.cloud_peak),
        ]:
            for v in arr:
                if v < 0.0:
                    raise ValueError(f"All values in {arr_name} must be non-negative.")
        for v in self.layer_depths:
            if v <= 0.0:
                raise ValueError("All layer_depths must be positive.")
        for v in self.cloud_sigma_x + self.cloud_sigma_y:
            if v <= 0.0:
                raise ValueError("All cloud sigmas must be positive.")
        if self.save_every < 1:
            raise ValueError("save_every must be at least 1.")


class CirrusCloudModelV2:
    def __init__(self, cfg: ModelConfig) -> None:
        cfg.validate()
        self.cfg = cfg

        self.x = np.arange(cfg.nx, dtype=np.float64) * cfg.dx
        self.y = np.arange(cfg.ny, dtype=np.float64) * cfg.dy
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.u, self.v = self._build_wind_fields()
        self.c = self._build_initial_cloud()
        self.source = self._build_source_fields()

        self.time_history: List[float] = []
        self.total_mass_history: List[float] = []
        self.layer_mass_history: List[List[float]] = [[] for _ in range(cfg.n_layers)]

        self.snapshots: List[np.ndarray] = []

        self._check_stability()

    def _build_wind_fields(self) -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)
        v = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)

        y_mid = 0.5 * self.y[-1]
        x_mid = 0.5 * self.x[-1]

        for k in range(self.cfg.n_layers):
            u[k] = self.cfg.u0[k] + self.cfg.shear_x[k] * (self.Y - y_mid)
            v[k] = self.cfg.v0[k] + self.cfg.shear_y[k] * (self.X - x_mid)

        return u, v

    def _build_initial_cloud(self) -> np.ndarray:
        c = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)

        for k in range(self.cfg.n_layers):
            dx0 = self.X - self.cfg.cloud_center_x
            dy0 = self.Y - self.cfg.cloud_center_y
            exponent = -(
                (dx0 * dx0) / (2.0 * self.cfg.cloud_sigma_x[k] ** 2)
                + (dy0 * dy0) / (2.0 * self.cfg.cloud_sigma_y[k] ** 2)
            )
            c[k] = self.cfg.cloud_peak[k] * np.exp(exponent)

        return c

    def _build_source_fields(self) -> np.ndarray:
        s = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)

        for k in range(self.cfg.n_layers):
            if self.cfg.source_strength[k] <= 0.0:
                continue
            dx0 = self.X - self.cfg.cloud_center_x
            dy0 = self.Y - self.cfg.cloud_center_y
            sx = 0.5 * self.cfg.cloud_sigma_x[k]
            sy = 0.5 * self.cfg.cloud_sigma_y[k]
            exponent = -((dx0 * dx0) / (2.0 * sx * sx) + (dy0 * dy0) / (2.0 * sy * sy))
            s[k] = self.cfg.source_strength[k] * np.exp(exponent)

        return s

    def _temperature_sublimation_rate(self, k: int) -> float:
        """
        Warmer temperatures imply faster sublimation.
        A simple synthetic relation around a cold cirrus regime.
        """
        tref = 220.0
        sensitivity = 0.04
        temp_factor = np.exp(sensitivity * (self.cfg.temperature[k] - tref))
        return self.cfg.base_sublimation[k] * temp_factor

    def _check_stability(self) -> None:
        max_cfl = 0.0
        max_diff = 0.0
        for k in range(self.cfg.n_layers):
            umax = float(np.max(np.abs(self.u[k])))
            vmax = float(np.max(np.abs(self.v[k])))
            cfl = max(umax * self.cfg.dt / self.cfg.dx, vmax * self.cfg.dt / self.cfg.dy)
            diff = (
                2.0 * self.cfg.kx[k] * self.cfg.dt / (self.cfg.dx ** 2)
                + 2.0 * self.cfg.ky[k] * self.cfg.dt / (self.cfg.dy ** 2)
            )
            max_cfl = max(max_cfl, cfl)
            max_diff = max(max_diff, diff)

        if max_cfl > 1.0:
            print(f"WARNING: maximum advection CFL is {max_cfl:.3f}", file=sys.stderr)
        if max_diff > 1.0:
            print(f"WARNING: maximum diffusion factor is {max_diff:.3f}", file=sys.stderr)

    def _apply_bc(self, a: np.ndarray) -> None:
        a[:, 0] = a[:, 1]
        a[:, -1] = a[:, -2]
        a[0, :] = a[1, :]
        a[-1, :] = a[-2, :]

    def _upwind_x(self, c: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = np.zeros_like(c)
        pos = u >= 0.0
        neg = ~pos

        out[:, 1:] = np.where(pos[:, 1:], (c[:, 1:] - c[:, :-1]) / self.cfg.dx, out[:, 1:])
        out[:, :-1] = np.where(neg[:, :-1], (c[:, 1:] - c[:, :-1]) / self.cfg.dx, out[:, :-1])

        out[:, 0] = out[:, 1]
        out[:, -1] = out[:, -2]
        return out

    def _upwind_y(self, c: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(c)
        pos = v >= 0.0
        neg = ~pos

        out[1:, :] = np.where(pos[1:, :], (c[1:, :] - c[:-1, :]) / self.cfg.dy, out[1:, :])
        out[:-1, :] = np.where(neg[:-1, :], (c[1:, :] - c[:-1, :]) / self.cfg.dy, out[:-1, :])

        out[0, :] = out[1, :]
        out[-1, :] = out[-2, :]
        return out

    def _laplacian(self, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d2x = np.zeros_like(c)
        d2y = np.zeros_like(c)

        d2x[:, 1:-1] = (c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]) / (self.cfg.dx ** 2)
        d2y[1:-1, :] = (c[2:, :] - 2.0 * c[1:-1, :] + c[:-2, :]) / (self.cfg.dy ** 2)

        d2x[:, 0] = d2x[:, 1]
        d2x[:, -1] = d2x[:, -2]
        d2y[0, :] = d2y[1, :]
        d2y[-1, :] = d2y[-2, :]
        return d2x, d2y

    def _vertical_exchange_tendency(self, c: np.ndarray) -> np.ndarray:
        """
        Simple symmetric exchange between neighboring layers.
        """
        alpha = self.cfg.exchange_updown
        exch = np.zeros_like(c)

        # lower
        exch[0] += alpha * (c[1] - c[0])

        # middle
        exch[1] += alpha * (c[0] - c[1]) + alpha * (c[2] - c[1])

        # upper
        exch[2] += alpha * (c[1] - c[2])

        return exch

    def step(self) -> None:
        c_new = np.zeros_like(self.c)
        exchange = self._vertical_exchange_tendency(self.c)

        for k in range(self.cfg.n_layers):
            dc_dx = self._upwind_x(self.c[k], self.u[k])
            dc_dy = self._upwind_y(self.c[k], self.v[k])
            d2x, d2y = self._laplacian(self.c[k])

            sub_rate = self._temperature_sublimation_rate(k)
            fall_rate = self.cfg.fall_speed[k] / self.cfg.layer_depths[k]

            tendency = (
                -self.u[k] * dc_dx
                -self.v[k] * dc_dy
                + self.cfg.kx[k] * d2x
                + self.cfg.ky[k] * d2y
                - sub_rate * self.c[k]
                - fall_rate * self.c[k]
                + exchange[k]
                + self.source[k]
            )

            c_new[k] = self.c[k] + self.cfg.dt * tendency
            c_new[k] = np.maximum(c_new[k], 0.0)
            self._apply_bc(c_new[k])

        self.c = c_new

    def _save_diagnostics(self, t: float) -> None:
        area = self.cfg.dx * self.cfg.dy
        layer_masses = [float(np.sum(self.c[k]) * area) for k in range(self.cfg.n_layers)]
        total_mass = float(sum(layer_masses))

        self.time_history.append(t)
        self.total_mass_history.append(total_mass)

        for k in range(self.cfg.n_layers):
            self.layer_mass_history[k].append(layer_masses[k])

        if len(self.snapshots) == 0 or (len(self.time_history) - 1) % self.cfg.save_every == 0:
            self.snapshots.append(np.sum(self.c, axis=0).copy())

    def run(self) -> None:
        for n in range(self.cfg.steps + 1):
            t = n * self.cfg.dt
            self._save_diagnostics(t)
            if n < self.cfg.steps:
                self.step()

    def total_field(self) -> np.ndarray:
        return np.sum(self.c, axis=0)

    def center_of_mass(self) -> Tuple[float, float]:
        tot = self.total_field()
        s = float(np.sum(tot))
        if s <= 0.0:
            return 0.0, 0.0
        xcm = float(np.sum(tot * self.X) / s)
        ycm = float(np.sum(tot * self.Y) / s)
        return xcm, ycm

    def summary(self) -> str:
        total = self.total_field()
        peak_idx = np.unravel_index(np.argmax(total), total.shape)
        ypk, xpk = int(peak_idx[0]), int(peak_idx[1])

        xcm, ycm = self.center_of_mass()

        lines = []
        lines.append("CIRRUS CLOUD MOTION MODEL V2 SUMMARY")
        lines.append("------------------------------------")
        lines.append(f"Simulation time          : {self.cfg.steps * self.cfg.dt / 3600.0:.2f} hours")
        lines.append(f"Grid                     : {self.cfg.nx} x {self.cfg.ny}")
        lines.append(f"Spacing                  : {self.cfg.dx:.1f} m x {self.cfg.dy:.1f} m")
        lines.append(f"Time step                : {self.cfg.dt:.1f} s")
        lines.append(f"Vertical exchange        : {self.cfg.exchange_updown:.3e} 1/s")
        lines.append("")

        for k in range(self.cfg.n_layers):
            initial_mass = self.layer_mass_history[k][0]
            final_mass = self.layer_mass_history[k][-1]
            pct = 100.0 * final_mass / initial_mass if initial_mass > 0.0 else 0.0
            lines.append(
                f"{self.cfg.layer_names[k]:<22}: "
                f"u={self.cfg.u0[k]:5.1f} m/s, "
                f"v={self.cfg.v0[k]:4.1f} m/s, "
                f"T={self.cfg.temperature[k]:5.1f} K, "
                f"fall={self.cfg.fall_speed[k]:.3f} m/s, "
                f"mass_retained={pct:6.2f}%"
            )

        lines.append("")
        lines.append(f"Initial total mass       : {self.total_mass_history[0]:.6e}")
        lines.append(f"Final total mass         : {self.total_mass_history[-1]:.6e}")
        lines.append(
            f"Total mass retained      : "
            f"{100.0 * self.total_mass_history[-1] / self.total_mass_history[0]:.2f}%"
        )
        lines.append(f"Peak total concentration : {float(np.max(total)):.6f}")
        lines.append(f"Peak location            : x={self.x[xpk] / 1000.0:.2f} km, y={self.y[ypk] / 1000.0:.2f} km")
        lines.append(f"Center of mass           : x={xcm / 1000.0:.2f} km, y={ycm / 1000.0:.2f} km")

        return "\n".join(lines)

    def plot(self) -> None:
        if not HAS_MPL:
            print("matplotlib not installed; skipping plots.")
            return

        extent = [self.x[0] / 1000.0, self.x[-1] / 1000.0, self.y[0] / 1000.0, self.y[-1] / 1000.0]

        fig1 = plt.figure(figsize=(10, 6))
        plt.imshow(self.total_field(), origin="lower", extent=extent, aspect="auto")
        plt.colorbar(label="Total cirrus concentration")
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.title("Final Total Cirrus Field")
        plt.tight_layout()

        fig2 = plt.figure(figsize=(10, 7))
        for k in range(self.cfg.n_layers):
            plt.plot(np.array(self.time_history) / 3600.0, self.layer_mass_history[k], label=self.cfg.layer_names[k])
        plt.plot(np.array(self.time_history) / 3600.0, self.total_mass_history, linewidth=2.0, label="Total")
        plt.xlabel("Time (hours)")
        plt.ylabel("Integrated mass proxy")
        plt.title("Layer and Total Cirrus Mass")
        plt.legend()
        plt.tight_layout()

        fig3 = plt.figure(figsize=(12, 4))
        for k in range(self.cfg.n_layers):
            plt.subplot(1, 3, k + 1)
            plt.imshow(self.c[k], origin="lower", extent=extent, aspect="auto")
            plt.title(self.cfg.layer_names[k])
            plt.xlabel("x (km)")
            if k == 0:
                plt.ylabel("y (km)")
        plt.tight_layout()

        plt.show()

    def animate_total_field(self, interval_ms: int = 120) -> None:
        if not HAS_MPL:
            print("matplotlib not installed; animation unavailable.")
            return
        if len(self.snapshots) < 2:
            print("Not enough snapshots for animation.")
            return

        extent = [self.x[0] / 1000.0, self.x[-1] / 1000.0, self.y[0] / 1000.0, self.y[-1] / 1000.0]

        fig, ax = plt.subplots(figsize=(10, 6))
        img = ax.imshow(self.snapshots[0], origin="lower", extent=extent, aspect="auto")
        plt.colorbar(img, ax=ax, label="Total cirrus concentration")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title("Cirrus Evolution")

        def update(frame: int):
            img.set_data(self.snapshots[frame])
            ax.set_title(f"Cirrus Evolution - frame {frame + 1}/{len(self.snapshots)}")
            return (img,)

        _anim = FuncAnimation(fig, update, frames=len(self.snapshots), interval=interval_ms, blit=False)
        plt.show()


def main() -> None:
    cfg = ModelConfig(
        nx=180,
        ny=120,
        dx=1000.0,
        dy=1000.0,
        dt=15.0,
        steps=960,  # 4 hours
        u0=(14.0, 22.0, 31.0),
        v0=(3.0, 5.0, 8.0),
        shear_x=(0.8e-4, 1.4e-4, 2.0e-4),
        shear_y=(-0.3e-4, -0.5e-4, -0.8e-4),
        kx=(80.0, 120.0, 180.0),
        ky=(60.0, 100.0, 140.0),
        temperature=(235.0, 225.0, 215.0),
        particle_radius=(70e-6, 45e-6, 25e-6),
        fall_speed=(0.050, 0.028, 0.012),
        base_sublimation=(3.0e-5, 1.8e-5, 1.0e-5),
        exchange_updown=2.0e-5,
        source_strength=(0.0, 0.0, 0.0),
        cloud_center_x=30000.0,
        cloud_center_y=35000.0,
        cloud_sigma_x=(9000.0, 12000.0, 15000.0),
        cloud_sigma_y=(6000.0, 8500.0, 11000.0),
        cloud_peak=(0.50, 0.85, 1.10),
        save_every=15,
    )

    model = CirrusCloudModelV2(cfg)
    model.run()
    print(model.summary())
    model.plot()

    # Uncomment to animate:
    # model.animate_total_field()


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
cirrus_cloud_motion_model_v2.py

A stronger Python 3 model of cirrus cloud motion.

FEATURES
--------
1. Three vertical cirrus layers:
      - lower cirrus layer
      - middle cirrus layer
      - upper cirrus layer

2. Separate horizontal winds in each layer.

3. Vertical exchange between adjacent layers.

4. Temperature-dependent sublimation:
      warmer layers lose cirrus faster.

5. Size-dependent sedimentation:
      larger representative ice particles fall faster.

6. Anisotropic turbulent diffusion.

7. Diagnostics:
      - total mass in each layer
      - domain-integrated total mass
      - center of mass
      - peak location

8. Optional animation.

GOVERNING IDEA
--------------
Each layer k has cloud concentration C_k(x,y,t) satisfying:

  dC_k/dt
    + u_k dC_k/dx + v_k dC_k/dy
    = Kx_k d2C_k/dx2 + Ky_k d2C_k/dy2
      - lambda_sub,k * C_k
      - lambda_fall,k * C_k
      + vertical_exchange
      + source_k

This is not a full microphysical cloud model, but it is stronger than a
single-layer bulk transport model and better suited to cirrus evolution.

DATA DICTIONARY
---------------
nx, ny
    Number of grid points in x and y.

dx, dy [m]
    Grid spacing.

dt [s]
    Time step.

steps
    Number of time steps.

n_layers
    Number of vertical layers, fixed here at 3.

layer_names
    Names of the cirrus layers.

layer_altitudes [m]
    Representative geometric heights of the layers.

layer_depths [m]
    Effective thickness of each layer.

u0[k], v0[k] [m/s]
    Base wind for layer k.

shear_x[k], shear_y[k] [1/s]
    Horizontal wind shear coefficients for layer k.

kx[k], ky[k] [m^2/s]
    Diffusion coefficients for layer k.

temperature[k] [K]
    Representative layer temperature.

particle_radius[k] [m]
    Representative ice particle radius for each layer.

fall_speed[k] [m/s]
    Effective sedimentation speed.

base_sublimation[k] [1/s]
    Baseline sublimation coefficient.

exchange_updown [1/s]
    Vertical exchange coefficient between neighboring layers.

source_strength[k]
    Source amplitude for each layer.

cloud_center_x, cloud_center_y [m]
    Initial cloud center.

cloud_sigma_x[k], cloud_sigma_y[k] [m]
    Initial Gaussian scale lengths per layer.

cloud_peak[k]
    Initial peak concentration for each layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import sys

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required.", file=sys.stderr)
    raise

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass(frozen=True)
class ModelConfig:
    nx: int = 180
    ny: int = 120
    dx: float = 1000.0
    dy: float = 1000.0
    dt: float = 15.0
    steps: int = 960

    n_layers: int = 3

    layer_names: Tuple[str, str, str] = ("Lower cirrus", "Middle cirrus", "Upper cirrus")
    layer_altitudes: Tuple[float, float, float] = (8000.0, 10000.0, 12000.0)
    layer_depths: Tuple[float, float, float] = (1500.0, 1800.0, 2200.0)

    u0: Tuple[float, float, float] = (14.0, 22.0, 31.0)
    v0: Tuple[float, float, float] = (3.0, 5.0, 8.0)

    shear_x: Tuple[float, float, float] = (0.8e-4, 1.4e-4, 2.0e-4)
    shear_y: Tuple[float, float, float] = (-0.3e-4, -0.5e-4, -0.8e-4)

    kx: Tuple[float, float, float] = (80.0, 120.0, 180.0)
    ky: Tuple[float, float, float] = (60.0, 100.0, 140.0)

    temperature: Tuple[float, float, float] = (235.0, 225.0, 215.0)

    particle_radius: Tuple[float, float, float] = (70e-6, 45e-6, 25e-6)
    fall_speed: Tuple[float, float, float] = (0.050, 0.028, 0.012)

    base_sublimation: Tuple[float, float, float] = (3.0e-5, 1.8e-5, 1.0e-5)

    exchange_updown: float = 2.0e-5

    source_strength: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    cloud_center_x: float = 30000.0
    cloud_center_y: float = 35000.0
    cloud_sigma_x: Tuple[float, float, float] = (9000.0, 12000.0, 15000.0)
    cloud_sigma_y: Tuple[float, float, float] = (6000.0, 8500.0, 11000.0)
    cloud_peak: Tuple[float, float, float] = (0.50, 0.85, 1.10)

    save_every: int = 20

    def validate(self) -> None:
        if self.n_layers != 3:
            raise ValueError("This version is implemented for exactly 3 layers.")
        if self.nx < 5 or self.ny < 5:
            raise ValueError("nx and ny must be at least 5.")
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError("dx and dy must be positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.steps < 1:
            raise ValueError("steps must be at least 1.")
        if self.exchange_updown < 0.0:
            raise ValueError("exchange_updown must be non-negative.")
        for arr_name, arr in [
            ("layer_depths", self.layer_depths),
            ("kx", self.kx),
            ("ky", self.ky),
            ("fall_speed", self.fall_speed),
            ("base_sublimation", self.base_sublimation),
            ("cloud_sigma_x", self.cloud_sigma_x),
            ("cloud_sigma_y", self.cloud_sigma_y),
            ("cloud_peak", self.cloud_peak),
        ]:
            for v in arr:
                if v < 0.0:
                    raise ValueError(f"All values in {arr_name} must be non-negative.")
        for v in self.layer_depths:
            if v <= 0.0:
                raise ValueError("All layer_depths must be positive.")
        for v in self.cloud_sigma_x + self.cloud_sigma_y:
            if v <= 0.0:
                raise ValueError("All cloud sigmas must be positive.")
        if self.save_every < 1:
            raise ValueError("save_every must be at least 1.")


class CirrusCloudModelV2:
    def __init__(self, cfg: ModelConfig) -> None:
        cfg.validate()
        self.cfg = cfg

        self.x = np.arange(cfg.nx, dtype=np.float64) * cfg.dx
        self.y = np.arange(cfg.ny, dtype=np.float64) * cfg.dy
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.u, self.v = self._build_wind_fields()
        self.c = self._build_initial_cloud()
        self.source = self._build_source_fields()

        self.time_history: List[float] = []
        self.total_mass_history: List[float] = []
        self.layer_mass_history: List[List[float]] = [[] for _ in range(cfg.n_layers)]

        self.snapshots: List[np.ndarray] = []

        self._check_stability()

    def _build_wind_fields(self) -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)
        v = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)

        y_mid = 0.5 * self.y[-1]
        x_mid = 0.5 * self.x[-1]

        for k in range(self.cfg.n_layers):
            u[k] = self.cfg.u0[k] + self.cfg.shear_x[k] * (self.Y - y_mid)
            v[k] = self.cfg.v0[k] + self.cfg.shear_y[k] * (self.X - x_mid)

        return u, v

    def _build_initial_cloud(self) -> np.ndarray:
        c = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)

        for k in range(self.cfg.n_layers):
            dx0 = self.X - self.cfg.cloud_center_x
            dy0 = self.Y - self.cfg.cloud_center_y
            exponent = -(
                (dx0 * dx0) / (2.0 * self.cfg.cloud_sigma_x[k] ** 2)
                + (dy0 * dy0) / (2.0 * self.cfg.cloud_sigma_y[k] ** 2)
            )
            c[k] = self.cfg.cloud_peak[k] * np.exp(exponent)

        return c

    def _build_source_fields(self) -> np.ndarray:
        s = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)

        for k in range(self.cfg.n_layers):
            if self.cfg.source_strength[k] <= 0.0:
                continue
            dx0 = self.X - self.cfg.cloud_center_x
            dy0 = self.Y - self.cfg.cloud_center_y
            sx = 0.5 * self.cfg.cloud_sigma_x[k]
            sy = 0.5 * self.cfg.cloud_sigma_y[k]
            exponent = -((dx0 * dx0) / (2.0 * sx * sx) + (dy0 * dy0) / (2.0 * sy * sy))
            s[k] = self.cfg.source_strength[k] * np.exp(exponent)

        return s

    def _temperature_sublimation_rate(self, k: int) -> float:
        """
        Warmer temperatures imply faster sublimation.
        A simple synthetic relation around a cold cirrus regime.
        """
        tref = 220.0
        sensitivity = 0.04
        temp_factor = np.exp(sensitivity * (self.cfg.temperature[k] - tref))
        return self.cfg.base_sublimation[k] * temp_factor

    def _check_stability(self) -> None:
        max_cfl = 0.0
        max_diff = 0.0
        for k in range(self.cfg.n_layers):
            umax = float(np.max(np.abs(self.u[k])))
            vmax = float(np.max(np.abs(self.v[k])))
            cfl = max(umax * self.cfg.dt / self.cfg.dx, vmax * self.cfg.dt / self.cfg.dy)
            diff = (
                2.0 * self.cfg.kx[k] * self.cfg.dt / (self.cfg.dx ** 2)
                + 2.0 * self.cfg.ky[k] * self.cfg.dt / (self.cfg.dy ** 2)
            )
            max_cfl = max(max_cfl, cfl)
            max_diff = max(max_diff, diff)

        if max_cfl > 1.0:
            print(f"WARNING: maximum advection CFL is {max_cfl:.3f}", file=sys.stderr)
        if max_diff > 1.0:
            print(f"WARNING: maximum diffusion factor is {max_diff:.3f}", file=sys.stderr)

    def _apply_bc(self, a: np.ndarray) -> None:
        a[:, 0] = a[:, 1]
        a[:, -1] = a[:, -2]
        a[0, :] = a[1, :]
        a[-1, :] = a[-2, :]

    def _upwind_x(self, c: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = np.zeros_like(c)
        pos = u >= 0.0
        neg = ~pos

        out[:, 1:] = np.where(pos[:, 1:], (c[:, 1:] - c[:, :-1]) / self.cfg.dx, out[:, 1:])
        out[:, :-1] = np.where(neg[:, :-1], (c[:, 1:] - c[:, :-1]) / self.cfg.dx, out[:, :-1])

        out[:, 0] = out[:, 1]
        out[:, -1] = out[:, -2]
        return out

    def _upwind_y(self, c: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(c)
        pos = v >= 0.0
        neg = ~pos

        out[1:, :] = np.where(pos[1:, :], (c[1:, :] - c[:-1, :]) / self.cfg.dy, out[1:, :])
        out[:-1, :] = np.where(neg[:-1, :], (c[1:, :] - c[:-1, :]) / self.cfg.dy, out[:-1, :])

        out[0, :] = out[1, :]
        out[-1, :] = out[-2, :]
        return out

    def _laplacian(self, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d2x = np.zeros_like(c)
        d2y = np.zeros_like(c)

        d2x[:, 1:-1] = (c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]) / (self.cfg.dx ** 2)
        d2y[1:-1, :] = (c[2:, :] - 2.0 * c[1:-1, :] + c[:-2, :]) / (self.cfg.dy ** 2)

        d2x[:, 0] = d2x[:, 1]
        d2x[:, -1] = d2x[:, -2]
        d2y[0, :] = d2y[1, :]
        d2y[-1, :] = d2y[-2, :]
        return d2x, d2y

    def _vertical_exchange_tendency(self, c: np.ndarray) -> np.ndarray:
        """
        Simple symmetric exchange between neighboring layers.
        """
        alpha = self.cfg.exchange_updown
        exch = np.zeros_like(c)

        # lower
        exch[0] += alpha * (c[1] - c[0])

        # middle
        exch[1] += alpha * (c[0] - c[1]) + alpha * (c[2] - c[1])

        # upper
        exch[2] += alpha * (c[1] - c[2])

        return exch

    def step(self) -> None:
        c_new = np.zeros_like(self.c)
        exchange = self._vertical_exchange_tendency(self.c)

        for k in range(self.cfg.n_layers):
            dc_dx = self._upwind_x(self.c[k], self.u[k])
            dc_dy = self._upwind_y(self.c[k], self.v[k])
            d2x, d2y = self._laplacian(self.c[k])

            sub_rate = self._temperature_sublimation_rate(k)
            fall_rate = self.cfg.fall_speed[k] / self.cfg.layer_depths[k]

            tendency = (
                -self.u[k] * dc_dx
                -self.v[k] * dc_dy
                + self.cfg.kx[k] * d2x
                + self.cfg.ky[k] * d2y
                - sub_rate * self.c[k]
                - fall_rate * self.c[k]
                + exchange[k]
                + self.source[k]
            )

            c_new[k] = self.c[k] + self.cfg.dt * tendency
            c_new[k] = np.maximum(c_new[k], 0.0)
            self._apply_bc(c_new[k])

        self.c = c_new

    def _save_diagnostics(self, t: float) -> None:
        area = self.cfg.dx * self.cfg.dy
        layer_masses = [float(np.sum(self.c[k]) * area) for k in range(self.cfg.n_layers)]
        total_mass = float(sum(layer_masses))

        self.time_history.append(t)
        self.total_mass_history.append(total_mass)

        for k in range(self.cfg.n_layers):
            self.layer_mass_history[k].append(layer_masses[k])

        if len(self.snapshots) == 0 or (len(self.time_history) - 1) % self.cfg.save_every == 0:
            self.snapshots.append(np.sum(self.c, axis=0).copy())

    def run(self) -> None:
        for n in range(self.cfg.steps + 1):
            t = n * self.cfg.dt
            self._save_diagnostics(t)
            if n < self.cfg.steps:
                self.step()

    def total_field(self) -> np.ndarray:
        return np.sum(self.c, axis=0)

    def center_of_mass(self) -> Tuple[float, float]:
        tot = self.total_field()
        s = float(np.sum(tot))
        if s <= 0.0:
            return 0.0, 0.0
        xcm = float(np.sum(tot * self.X) / s)
        ycm = float(np.sum(tot * self.Y) / s)
        return xcm, ycm

    def summary(self) -> str:
        total = self.total_field()
        peak_idx = np.unravel_index(np.argmax(total), total.shape)
        ypk, xpk = int(peak_idx[0]), int(peak_idx[1])

        xcm, ycm = self.center_of_mass()

        lines = []
        lines.append("CIRRUS CLOUD MOTION MODEL V2 SUMMARY")
        lines.append("------------------------------------")
        lines.append(f"Simulation time          : {self.cfg.steps * self.cfg.dt / 3600.0:.2f} hours")
        lines.append(f"Grid                     : {self.cfg.nx} x {self.cfg.ny}")
        lines.append(f"Spacing                  : {self.cfg.dx:.1f} m x {self.cfg.dy:.1f} m")
        lines.append(f"Time step                : {self.cfg.dt:.1f} s")
        lines.append(f"Vertical exchange        : {self.cfg.exchange_updown:.3e} 1/s")
        lines.append("")

        for k in range(self.cfg.n_layers):
            initial_mass = self.layer_mass_history[k][0]
            final_mass = self.layer_mass_history[k][-1]
            pct = 100.0 * final_mass / initial_mass if initial_mass > 0.0 else 0.0
            lines.append(
                f"{self.cfg.layer_names[k]:<22}: "
                f"u={self.cfg.u0[k]:5.1f} m/s, "
                f"v={self.cfg.v0[k]:4.1f} m/s, "
                f"T={self.cfg.temperature[k]:5.1f} K, "
                f"fall={self.cfg.fall_speed[k]:.3f} m/s, "
                f"mass_retained={pct:6.2f}%"
            )

        lines.append("")
        lines.append(f"Initial total mass       : {self.total_mass_history[0]:.6e}")
        lines.append(f"Final total mass         : {self.total_mass_history[-1]:.6e}")
        lines.append(
            f"Total mass retained      : "
            f"{100.0 * self.total_mass_history[-1] / self.total_mass_history[0]:.2f}%"
        )
        lines.append(f"Peak total concentration : {float(np.max(total)):.6f}")
        lines.append(f"Peak location            : x={self.x[xpk] / 1000.0:.2f} km, y={self.y[ypk] / 1000.0:.2f} km")
        lines.append(f"Center of mass           : x={xcm / 1000.0:.2f} km, y={ycm / 1000.0:.2f} km")

        return "\n".join(lines)

    def plot(self) -> None:
        if not HAS_MPL:
            print("matplotlib not installed; skipping plots.")
            return

        extent = [self.x[0] / 1000.0, self.x[-1] / 1000.0, self.y[0] / 1000.0, self.y[-1] / 1000.0]

        fig1 = plt.figure(figsize=(10, 6))
        plt.imshow(self.total_field(), origin="lower", extent=extent, aspect="auto")
        plt.colorbar(label="Total cirrus concentration")
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.title("Final Total Cirrus Field")
        plt.tight_layout()

        fig2 = plt.figure(figsize=(10, 7))
        for k in range(self.cfg.n_layers):
            plt.plot(np.array(self.time_history) / 3600.0, self.layer_mass_history[k], label=self.cfg.layer_names[k])
        plt.plot(np.array(self.time_history) / 3600.0, self.total_mass_history, linewidth=2.0, label="Total")
        plt.xlabel("Time (hours)")
        plt.ylabel("Integrated mass proxy")
        plt.title("Layer and Total Cirrus Mass")
        plt.legend()
        plt.tight_layout()

        fig3 = plt.figure(figsize=(12, 4))
        for k in range(self.cfg.n_layers):
            plt.subplot(1, 3, k + 1)
            plt.imshow(self.c[k], origin="lower", extent=extent, aspect="auto")
            plt.title(self.cfg.layer_names[k])
            plt.xlabel("x (km)")
            if k == 0:
                plt.ylabel("y (km)")
        plt.tight_layout()

        plt.show()

    def animate_total_field(self, interval_ms: int = 120) -> None:
        if not HAS_MPL:
            print("matplotlib not installed; animation unavailable.")
            return
        if len(self.snapshots) < 2:
            print("Not enough snapshots for animation.")
            return

        extent = [self.x[0] / 1000.0, self.x[-1] / 1000.0, self.y[0] / 1000.0, self.y[-1] / 1000.0]

        fig, ax = plt.subplots(figsize=(10, 6))
        img = ax.imshow(self.snapshots[0], origin="lower", extent=extent, aspect="auto")
        plt.colorbar(img, ax=ax, label="Total cirrus concentration")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title("Cirrus Evolution")

        def update(frame: int):
            img.set_data(self.snapshots[frame])
            ax.set_title(f"Cirrus Evolution - frame {frame + 1}/{len(self.snapshots)}")
            return (img,)

        _anim = FuncAnimation(fig, update, frames=len(self.snapshots), interval=interval_ms, blit=False)
        plt.show()


def main() -> None:
    cfg = ModelConfig(
        nx=180,
        ny=120,
        dx=1000.0,
        dy=1000.0,
        dt=15.0,
        steps=960,  # 4 hours
        u0=(14.0, 22.0, 31.0),
        v0=(3.0, 5.0, 8.0),
        shear_x=(0.8e-4, 1.4e-4, 2.0e-4),
        shear_y=(-0.3e-4, -0.5e-4, -0.8e-4),
        kx=(80.0, 120.0, 180.0),
        ky=(60.0, 100.0, 140.0),
        temperature=(235.0, 225.0, 215.0),
        particle_radius=(70e-6, 45e-6, 25e-6),
        fall_speed=(0.050, 0.028, 0.012),
        base_sublimation=(3.0e-5, 1.8e-5, 1.0e-5),
        exchange_updown=2.0e-5,
        source_strength=(0.0, 0.0, 0.0),
        cloud_center_x=30000.0,
        cloud_center_y=35000.0,
        cloud_sigma_x=(9000.0, 12000.0, 15000.0),
        cloud_sigma_y=(6000.0, 8500.0, 11000.0),
        cloud_peak=(0.50, 0.85, 1.10),
        save_every=15,
    )

    model = CirrusCloudModelV2(cfg)
    model.run()
    print(model.summary())
    model.plot()

    # Uncomment to animate:
    # model.animate_total_field()


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
cirrus_cloud_motion_model_v2.py

A stronger Python 3 model of cirrus cloud motion.

FEATURES
--------
1. Three vertical cirrus layers:
      - lower cirrus layer
      - middle cirrus layer
      - upper cirrus layer

2. Separate horizontal winds in each layer.

3. Vertical exchange between adjacent layers.

4. Temperature-dependent sublimation:
      warmer layers lose cirrus faster.

5. Size-dependent sedimentation:
      larger representative ice particles fall faster.

6. Anisotropic turbulent diffusion.

7. Diagnostics:
      - total mass in each layer
      - domain-integrated total mass
      - center of mass
      - peak location

8. Optional animation.

GOVERNING IDEA
--------------
Each layer k has cloud concentration C_k(x,y,t) satisfying:

  dC_k/dt
    + u_k dC_k/dx + v_k dC_k/dy
    = Kx_k d2C_k/dx2 + Ky_k d2C_k/dy2
      - lambda_sub,k * C_k
      - lambda_fall,k * C_k
      + vertical_exchange
      + source_k

This is not a full microphysical cloud model, but it is stronger than a
single-layer bulk transport model and better suited to cirrus evolution.

DATA DICTIONARY
---------------
nx, ny
    Number of grid points in x and y.

dx, dy [m]
    Grid spacing.

dt [s]
    Time step.

steps
    Number of time steps.

n_layers
    Number of vertical layers, fixed here at 3.

layer_names
    Names of the cirrus layers.

layer_altitudes [m]
    Representative geometric heights of the layers.

layer_depths [m]
    Effective thickness of each layer.

u0[k], v0[k] [m/s]
    Base wind for layer k.

shear_x[k], shear_y[k] [1/s]
    Horizontal wind shear coefficients for layer k.

kx[k], ky[k] [m^2/s]
    Diffusion coefficients for layer k.

temperature[k] [K]
    Representative layer temperature.

particle_radius[k] [m]
    Representative ice particle radius for each layer.

fall_speed[k] [m/s]
    Effective sedimentation speed.

base_sublimation[k] [1/s]
    Baseline sublimation coefficient.

exchange_updown [1/s]
    Vertical exchange coefficient between neighboring layers.

source_strength[k]
    Source amplitude for each layer.

cloud_center_x, cloud_center_y [m]
    Initial cloud center.

cloud_sigma_x[k], cloud_sigma_y[k] [m]
    Initial Gaussian scale lengths per layer.

cloud_peak[k]
    Initial peak concentration for each layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import sys

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required.", file=sys.stderr)
    raise

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass(frozen=True)
class ModelConfig:
    nx: int = 180
    ny: int = 120
    dx: float = 1000.0
    dy: float = 1000.0
    dt: float = 15.0
    steps: int = 960

    n_layers: int = 3

    layer_names: Tuple[str, str, str] = ("Lower cirrus", "Middle cirrus", "Upper cirrus")
    layer_altitudes: Tuple[float, float, float] = (8000.0, 10000.0, 12000.0)
    layer_depths: Tuple[float, float, float] = (1500.0, 1800.0, 2200.0)

    u0: Tuple[float, float, float] = (14.0, 22.0, 31.0)
    v0: Tuple[float, float, float] = (3.0, 5.0, 8.0)

    shear_x: Tuple[float, float, float] = (0.8e-4, 1.4e-4, 2.0e-4)
    shear_y: Tuple[float, float, float] = (-0.3e-4, -0.5e-4, -0.8e-4)

    kx: Tuple[float, float, float] = (80.0, 120.0, 180.0)
    ky: Tuple[float, float, float] = (60.0, 100.0, 140.0)

    temperature: Tuple[float, float, float] = (235.0, 225.0, 215.0)

    particle_radius: Tuple[float, float, float] = (70e-6, 45e-6, 25e-6)
    fall_speed: Tuple[float, float, float] = (0.050, 0.028, 0.012)

    base_sublimation: Tuple[float, float, float] = (3.0e-5, 1.8e-5, 1.0e-5)

    exchange_updown: float = 2.0e-5

    source_strength: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    cloud_center_x: float = 30000.0
    cloud_center_y: float = 35000.0
    cloud_sigma_x: Tuple[float, float, float] = (9000.0, 12000.0, 15000.0)
    cloud_sigma_y: Tuple[float, float, float] = (6000.0, 8500.0, 11000.0)
    cloud_peak: Tuple[float, float, float] = (0.50, 0.85, 1.10)

    save_every: int = 20

    def validate(self) -> None:
        if self.n_layers != 3:
            raise ValueError("This version is implemented for exactly 3 layers.")
        if self.nx < 5 or self.ny < 5:
            raise ValueError("nx and ny must be at least 5.")
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError("dx and dy must be positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.steps < 1:
            raise ValueError("steps must be at least 1.")
        if self.exchange_updown < 0.0:
            raise ValueError("exchange_updown must be non-negative.")
        for arr_name, arr in [
            ("layer_depths", self.layer_depths),
            ("kx", self.kx),
            ("ky", self.ky),
            ("fall_speed", self.fall_speed),
            ("base_sublimation", self.base_sublimation),
            ("cloud_sigma_x", self.cloud_sigma_x),
            ("cloud_sigma_y", self.cloud_sigma_y),
            ("cloud_peak", self.cloud_peak),
        ]:
            for v in arr:
                if v < 0.0:
                    raise ValueError(f"All values in {arr_name} must be non-negative.")
        for v in self.layer_depths:
            if v <= 0.0:
                raise ValueError("All layer_depths must be positive.")
        for v in self.cloud_sigma_x + self.cloud_sigma_y:
            if v <= 0.0:
                raise ValueError("All cloud sigmas must be positive.")
        if self.save_every < 1:
            raise ValueError("save_every must be at least 1.")


class CirrusCloudModelV2:
    def __init__(self, cfg: ModelConfig) -> None:
        cfg.validate()
        self.cfg = cfg

        self.x = np.arange(cfg.nx, dtype=np.float64) * cfg.dx
        self.y = np.arange(cfg.ny, dtype=np.float64) * cfg.dy
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.u, self.v = self._build_wind_fields()
        self.c = self._build_initial_cloud()
        self.source = self._build_source_fields()

        self.time_history: List[float] = []
        self.total_mass_history: List[float] = []
        self.layer_mass_history: List[List[float]] = [[] for _ in range(cfg.n_layers)]

        self.snapshots: List[np.ndarray] = []

        self._check_stability()

    def _build_wind_fields(self) -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)
        v = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)

        y_mid = 0.5 * self.y[-1]
        x_mid = 0.5 * self.x[-1]

        for k in range(self.cfg.n_layers):
            u[k] = self.cfg.u0[k] + self.cfg.shear_x[k] * (self.Y - y_mid)
            v[k] = self.cfg.v0[k] + self.cfg.shear_y[k] * (self.X - x_mid)

        return u, v

    def _build_initial_cloud(self) -> np.ndarray:
        c = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)

        for k in range(self.cfg.n_layers):
            dx0 = self.X - self.cfg.cloud_center_x
            dy0 = self.Y - self.cfg.cloud_center_y
            exponent = -(
                (dx0 * dx0) / (2.0 * self.cfg.cloud_sigma_x[k] ** 2)
                + (dy0 * dy0) / (2.0 * self.cfg.cloud_sigma_y[k] ** 2)
            )
            c[k] = self.cfg.cloud_peak[k] * np.exp(exponent)

        return c

    def _build_source_fields(self) -> np.ndarray:
        s = np.zeros((self.cfg.n_layers, self.cfg.ny, self.cfg.nx), dtype=np.float64)

        for k in range(self.cfg.n_layers):
            if self.cfg.source_strength[k] <= 0.0:
                continue
            dx0 = self.X - self.cfg.cloud_center_x
            dy0 = self.Y - self.cfg.cloud_center_y
            sx = 0.5 * self.cfg.cloud_sigma_x[k]
            sy = 0.5 * self.cfg.cloud_sigma_y[k]
            exponent = -((dx0 * dx0) / (2.0 * sx * sx) + (dy0 * dy0) / (2.0 * sy * sy))
            s[k] = self.cfg.source_strength[k] * np.exp(exponent)

        return s

    def _temperature_sublimation_rate(self, k: int) -> float:
        """
        Warmer temperatures imply faster sublimation.
        A simple synthetic relation around a cold cirrus regime.
        """
        tref = 220.0
        sensitivity = 0.04
        temp_factor = np.exp(sensitivity * (self.cfg.temperature[k] - tref))
        return self.cfg.base_sublimation[k] * temp_factor

    def _check_stability(self) -> None:
        max_cfl = 0.0
        max_diff = 0.0
        for k in range(self.cfg.n_layers):
            umax = float(np.max(np.abs(self.u[k])))
            vmax = float(np.max(np.abs(self.v[k])))
            cfl = max(umax * self.cfg.dt / self.cfg.dx, vmax * self.cfg.dt / self.cfg.dy)
            diff = (
                2.0 * self.cfg.kx[k] * self.cfg.dt / (self.cfg.dx ** 2)
                + 2.0 * self.cfg.ky[k] * self.cfg.dt / (self.cfg.dy ** 2)
            )
            max_cfl = max(max_cfl, cfl)
            max_diff = max(max_diff, diff)

        if max_cfl > 1.0:
            print(f"WARNING: maximum advection CFL is {max_cfl:.3f}", file=sys.stderr)
        if max_diff > 1.0:
            print(f"WARNING: maximum diffusion factor is {max_diff:.3f}", file=sys.stderr)

    def _apply_bc(self, a: np.ndarray) -> None:
        a[:, 0] = a[:, 1]
        a[:, -1] = a[:, -2]
        a[0, :] = a[1, :]
        a[-1, :] = a[-2, :]

    def _upwind_x(self, c: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = np.zeros_like(c)
        pos = u >= 0.0
        neg = ~pos

        out[:, 1:] = np.where(pos[:, 1:], (c[:, 1:] - c[:, :-1]) / self.cfg.dx, out[:, 1:])
        out[:, :-1] = np.where(neg[:, :-1], (c[:, 1:] - c[:, :-1]) / self.cfg.dx, out[:, :-1])

        out[:, 0] = out[:, 1]
        out[:, -1] = out[:, -2]
        return out

    def _upwind_y(self, c: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(c)
        pos = v >= 0.0
        neg = ~pos

        out[1:, :] = np.where(pos[1:, :], (c[1:, :] - c[:-1, :]) / self.cfg.dy, out[1:, :])
        out[:-1, :] = np.where(neg[:-1, :], (c[1:, :] - c[:-1, :]) / self.cfg.dy, out[:-1, :])

        out[0, :] = out[1, :]
        out[-1, :] = out[-2, :]
        return out

    def _laplacian(self, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d2x = np.zeros_like(c)
        d2y = np.zeros_like(c)

        d2x[:, 1:-1] = (c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]) / (self.cfg.dx ** 2)
        d2y[1:-1, :] = (c[2:, :] - 2.0 * c[1:-1, :] + c[:-2, :]) / (self.cfg.dy ** 2)

        d2x[:, 0] = d2x[:, 1]
        d2x[:, -1] = d2x[:, -2]
        d2y[0, :] = d2y[1, :]
        d2y[-1, :] = d2y[-2, :]
        return d2x, d2y

    def _vertical_exchange_tendency(self, c: np.ndarray) -> np.ndarray:
        """
        Simple symmetric exchange between neighboring layers.
        """
        alpha = self.cfg.exchange_updown
        exch = np.zeros_like(c)

        # lower
        exch[0] += alpha * (c[1] - c[0])

        # middle
        exch[1] += alpha * (c[0] - c[1]) + alpha * (c[2] - c[1])

        # upper
        exch[2] += alpha * (c[1] - c[2])

        return exch

    def step(self) -> None:
        c_new = np.zeros_like(self.c)
        exchange = self._vertical_exchange_tendency(self.c)

        for k in range(self.cfg.n_layers):
            dc_dx = self._upwind_x(self.c[k], self.u[k])
            dc_dy = self._upwind_y(self.c[k], self.v[k])
            d2x, d2y = self._laplacian(self.c[k])

            sub_rate = self._temperature_sublimation_rate(k)
            fall_rate = self.cfg.fall_speed[k] / self.cfg.layer_depths[k]

            tendency = (
                -self.u[k] * dc_dx
                -self.v[k] * dc_dy
                + self.cfg.kx[k] * d2x
                + self.cfg.ky[k] * d2y
                - sub_rate * self.c[k]
                - fall_rate * self.c[k]
                + exchange[k]
                + self.source[k]
            )

            c_new[k] = self.c[k] + self.cfg.dt * tendency
            c_new[k] = np.maximum(c_new[k], 0.0)
            self._apply_bc(c_new[k])

        self.c = c_new

    def _save_diagnostics(self, t: float) -> None:
        area = self.cfg.dx * self.cfg.dy
        layer_masses = [float(np.sum(self.c[k]) * area) for k in range(self.cfg.n_layers)]
        total_mass = float(sum(layer_masses))

        self.time_history.append(t)
        self.total_mass_history.append(total_mass)

        for k in range(self.cfg.n_layers):
            self.layer_mass_history[k].append(layer_masses[k])

        if len(self.snapshots) == 0 or (len(self.time_history) - 1) % self.cfg.save_every == 0:
            self.snapshots.append(np.sum(self.c, axis=0).copy())

    def run(self) -> None:
        for n in range(self.cfg.steps + 1):
            t = n * self.cfg.dt
            self._save_diagnostics(t)
            if n < self.cfg.steps:
                self.step()

    def total_field(self) -> np.ndarray:
        return np.sum(self.c, axis=0)

    def center_of_mass(self) -> Tuple[float, float]:
        tot = self.total_field()
        s = float(np.sum(tot))
        if s <= 0.0:
            return 0.0, 0.0
        xcm = float(np.sum(tot * self.X) / s)
        ycm = float(np.sum(tot * self.Y) / s)
        return xcm, ycm

    def summary(self) -> str:
        total = self.total_field()
        peak_idx = np.unravel_index(np.argmax(total), total.shape)
        ypk, xpk = int(peak_idx[0]), int(peak_idx[1])

        xcm, ycm = self.center_of_mass()

        lines = []
        lines.append("CIRRUS CLOUD MOTION MODEL V2 SUMMARY")
        lines.append("------------------------------------")
        lines.append(f"Simulation time          : {self.cfg.steps * self.cfg.dt / 3600.0:.2f} hours")
        lines.append(f"Grid                     : {self.cfg.nx} x {self.cfg.ny}")
        lines.append(f"Spacing                  : {self.cfg.dx:.1f} m x {self.cfg.dy:.1f} m")
        lines.append(f"Time step                : {self.cfg.dt:.1f} s")
        lines.append(f"Vertical exchange        : {self.cfg.exchange_updown:.3e} 1/s")
        lines.append("")

        for k in range(self.cfg.n_layers):
            initial_mass = self.layer_mass_history[k][0]
            final_mass = self.layer_mass_history[k][-1]
            pct = 100.0 * final_mass / initial_mass if initial_mass > 0.0 else 0.0
            lines.append(
                f"{self.cfg.layer_names[k]:<22}: "
                f"u={self.cfg.u0[k]:5.1f} m/s, "
                f"v={self.cfg.v0[k]:4.1f} m/s, "
                f"T={self.cfg.temperature[k]:5.1f} K, "
                f"fall={self.cfg.fall_speed[k]:.3f} m/s, "
                f"mass_retained={pct:6.2f}%"
            )

        lines.append("")
        lines.append(f"Initial total mass       : {self.total_mass_history[0]:.6e}")
        lines.append(f"Final total mass         : {self.total_mass_history[-1]:.6e}")
        lines.append(
            f"Total mass retained      : "
            f"{100.0 * self.total_mass_history[-1] / self.total_mass_history[0]:.2f}%"
        )
        lines.append(f"Peak total concentration : {float(np.max(total)):.6f}")
        lines.append(f"Peak location            : x={self.x[xpk] / 1000.0:.2f} km, y={self.y[ypk] / 1000.0:.2f} km")
        lines.append(f"Center of mass           : x={xcm / 1000.0:.2f} km, y={ycm / 1000.0:.2f} km")

        return "\n".join(lines)

    def plot(self) -> None:
        if not HAS_MPL:
            print("matplotlib not installed; skipping plots.")
            return

        extent = [self.x[0] / 1000.0, self.x[-1] / 1000.0, self.y[0] / 1000.0, self.y[-1] / 1000.0]

        fig1 = plt.figure(figsize=(10, 6))
        plt.imshow(self.total_field(), origin="lower", extent=extent, aspect="auto")
        plt.colorbar(label="Total cirrus concentration")
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.title("Final Total Cirrus Field")
        plt.tight_layout()

        fig2 = plt.figure(figsize=(10, 7))
        for k in range(self.cfg.n_layers):
            plt.plot(np.array(self.time_history) / 3600.0, self.layer_mass_history[k], label=self.cfg.layer_names[k])
        plt.plot(np.array(self.time_history) / 3600.0, self.total_mass_history, linewidth=2.0, label="Total")
        plt.xlabel("Time (hours)")
        plt.ylabel("Integrated mass proxy")
        plt.title("Layer and Total Cirrus Mass")
        plt.legend()
        plt.tight_layout()

        fig3 = plt.figure(figsize=(12, 4))
        for k in range(self.cfg.n_layers):
            plt.subplot(1, 3, k + 1)
            plt.imshow(self.c[k], origin="lower", extent=extent, aspect="auto")
            plt.title(self.cfg.layer_names[k])
            plt.xlabel("x (km)")
            if k == 0:
                plt.ylabel("y (km)")
        plt.tight_layout()

        plt.show()

    def animate_total_field(self, interval_ms: int = 120) -> None:
        if not HAS_MPL:
            print("matplotlib not installed; animation unavailable.")
            return
        if len(self.snapshots) < 2:
            print("Not enough snapshots for animation.")
            return

        extent = [self.x[0] / 1000.0, self.x[-1] / 1000.0, self.y[0] / 1000.0, self.y[-1] / 1000.0]

        fig, ax = plt.subplots(figsize=(10, 6))
        img = ax.imshow(self.snapshots[0], origin="lower", extent=extent, aspect="auto")
        plt.colorbar(img, ax=ax, label="Total cirrus concentration")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title("Cirrus Evolution")

        def update(frame: int):
            img.set_data(self.snapshots[frame])
            ax.set_title(f"Cirrus Evolution - frame {frame + 1}/{len(self.snapshots)}")
            return (img,)

        _anim = FuncAnimation(fig, update, frames=len(self.snapshots), interval=interval_ms, blit=False)
        plt.show()


def main() -> None:
    cfg = ModelConfig(
        nx=180,
        ny=120,
        dx=1000.0,
        dy=1000.0,
        dt=15.0,
        steps=960,  # 4 hours
        u0=(14.0, 22.0, 31.0),
        v0=(3.0, 5.0, 8.0),
        shear_x=(0.8e-4, 1.4e-4, 2.0e-4),
        shear_y=(-0.3e-4, -0.5e-4, -0.8e-4),
        kx=(80.0, 120.0, 180.0),
        ky=(60.0, 100.0, 140.0),
        temperature=(235.0, 225.0, 215.0),
        particle_radius=(70e-6, 45e-6, 25e-6),
        fall_speed=(0.050, 0.028, 0.012),
        base_sublimation=(3.0e-5, 1.8e-5, 1.0e-5),
        exchange_updown=2.0e-5,
        source_strength=(0.0, 0.0, 0.0),
        cloud_center_x=30000.0,
        cloud_center_y=35000.0,
        cloud_sigma_x=(9000.0, 12000.0, 15000.0),
        cloud_sigma_y=(6000.0, 8500.0, 11000.0),
        cloud_peak=(0.50, 0.85, 1.10),
        save_every=15,
    )

    model = CirrusCloudModelV2(cfg)
    model.run()
    print(model.summary())
    model.plot()

    # Uncomment to animate:
    # model.animate_total_field()


if __name__ == "__main__":
    main()
