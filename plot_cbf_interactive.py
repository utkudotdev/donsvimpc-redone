"""
Interactive heatmap of the CBF over (x, y) for a chosen environment.

Sliders/checkboxes set the remaining state dimensions (v, theta, per-obstacle
alpha, per-obstacle forward); the heatmap recomputes when a slider is released
or a checkbox/radio button is clicked.

When --ncbf is provided the radio buttons let you switch the plotted value
between:
    - max(h, V): the full NCBF (max of analytic h and the network)
    - h:         the analytic h_fn only
    - V:         the NCBF network only

Usage:
    python plot_cbf_interactive.py [--env basic] [--ncbf path/to/ckpt] [--resolution 100]
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.widgets import CheckButtons, RadioButtons, Slider

from networks.feature import make_dubins_features

# Fixed color scale: keeps comparisons across state changes meaningful, and
# matches the heatmap output by train.py's visualize_ncbf.
VMIN = -1.0
VMAX = 1.5
CMAP = "RdBu_r"

from dynamics.environment_dynamics import State
from dynamics.dubins_dynamics import DubinsState
from dynamics.obstacle_dynamics import ObstacleState
from environments.dubins import get_environment_parameters
from networks.ncbf import NCBF, NCBFNetwork, load_checkpoint
from tasks.dubins import compute_h_vector


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Interactively visualize the CBF over (x, y) for a chosen environment."
    )
    parser.add_argument(
        "--env", type=str, default="basic", help="Environment name (default: basic)"
    )
    parser.add_argument(
        "--ncbf",
        type=Path,
        default=None,
        help="Optional NCBF checkpoint directory. Enables max(h,V) / h / V plotting modes.",
    )
    parser.add_argument(
        "--resolution", type=int, default=100, help="Grid bins per axis (default: 100)"
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    params = get_environment_parameters(args.env)

    h_fn = compute_h_vector

    use_ncbf = args.ncbf is not None
    ncbf_combined = None
    ncbf_network = None
    if use_ncbf:
        print(f"Loading NCBF network from {args.ncbf}")
        ncbf_network = load_checkpoint(args.ncbf)[0]
        ncbf_combined = NCBF(h_fn=h_fn, ncbf_network=ncbf_network)

    num_obstacles = int(params.obstacle_params.radius.shape[0])

    grid_x = jnp.linspace(params.x_min, params.x_max, args.resolution)
    grid_y = jnp.linspace(params.y_min, params.y_max, args.resolution)
    X, Y = jnp.meshgrid(grid_x, grid_y)

    def make_grid_fn(value_fn):
        @jax.jit
        def compute(v, theta, alphas, forwards):
            obs = ObstacleState(alpha=alphas, forward=forwards)

            def at_xy(x, y):
                s = State(
                    dubins_state=DubinsState(x=x, y=y, v=v, theta=theta),
                    obstacle_state=obs,
                )
                return jnp.max(value_fn(s, params))

            return jax.vmap(jax.vmap(at_xy, in_axes=(0, 0)), in_axes=(0, 0))(X, Y)

        return compute

    def network_only(s: State, p):
        return ncbf_network(make_dubins_features(s, p))

    grid_fns: dict[str, callable] = {"h": make_grid_fn(h_fn)}
    if use_ncbf:
        grid_fns["max(h, V)"] = make_grid_fn(ncbf_combined)
        grid_fns["V"] = make_grid_fn(network_only)

    mode_options = ["max(h, V)", "h", "V"] if use_ncbf else ["h"]
    initial_mode = mode_options[0]

    @jax.jit
    def obstacle_positions(alphas):
        obs = ObstacleState(alpha=alphas, forward=jnp.ones_like(alphas, dtype=bool))
        return jax.vmap(ObstacleState.position)(obs, params.obstacle_params)

    v0 = 0.1
    theta0 = float(jnp.pi / 2.0)
    alpha0 = np.zeros(num_obstacles, dtype=np.float32)
    forward0 = np.ones(num_obstacles, dtype=bool)

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle(f"Interactive CBF heatmap — env={args.env}")

    ax = fig.add_axes([0.07, 0.34, 0.52, 0.58])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(params.x_min), float(params.x_max))
    ax.set_ylim(float(params.y_min), float(params.y_max))

    initial_grid = np.asarray(
        grid_fns[initial_mode](
            jnp.array(v0),
            jnp.array(theta0),
            jnp.asarray(alpha0),
            jnp.asarray(forward0),
        )
    )

    x_centers = np.asarray(grid_x)
    y_centers = np.asarray(grid_y)

    norm = TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=VMAX)
    mesh = ax.pcolormesh(
        x_centers,
        y_centers,
        initial_grid,
        cmap=CMAP,
        shading="nearest",
        norm=norm,
    )
    fig.colorbar(mesh, ax=ax, label=f"value (>0 ⇒ unsafe; clipped to [{VMIN}, {VMAX}])")

    title = ax.set_title(f"mode={initial_mode}")

    obs_init = np.asarray(obstacle_positions(jnp.asarray(alpha0)))
    radii = np.asarray(params.obstacle_params.radius)
    obstacle_patches = []
    for k in range(num_obstacles):
        circle = patches.Circle(
            (obs_init[k, 0], obs_init[k, 1]),
            radius=float(radii[k]),
            edgecolor="black",
            facecolor="none",
            linewidth=1.5,
        )
        ax.add_patch(circle)
        obstacle_patches.append(circle)

    # --- Sliders ---
    slider_left = 0.10
    slider_width = 0.50
    slider_height = 0.025
    slider_axes = []
    sliders: dict[str, Slider] = {}

    def add_slider(label, valmin, valmax, valinit, idx):
        bottom = 0.24 - idx * 0.035
        ax_s = fig.add_axes([slider_left, bottom, slider_width, slider_height])
        slider = Slider(ax_s, label, valmin, valmax, valinit=valinit)
        slider_axes.append(ax_s)
        return slider

    sliders["v"] = add_slider(
        "v",
        float(params.dubins_params.velocity_min),
        float(params.dubins_params.velocity_max),
        v0,
        0,
    )
    sliders["theta"] = add_slider("theta", -float(jnp.pi), float(jnp.pi), theta0, 1)
    for k in range(num_obstacles):
        sliders[f"alpha_{k}"] = add_slider(
            f"alpha_{k}", 0.0, 1.0, float(alpha0[k]), 2 + k
        )

    # --- CheckButtons for forward (right side) ---
    ax_check = fig.add_axes([0.65, 0.55, 0.13, 0.30])
    ax_check.set_title("forward", fontsize=10)
    check_labels = [f"obs_{k}" for k in range(num_obstacles)]
    check_init = [bool(forward0[k]) for k in range(num_obstacles)]
    checks = CheckButtons(ax_check, check_labels, check_init)

    # --- RadioButtons for plotting mode (only when there is a choice) ---
    radio = None
    if len(mode_options) > 1:
        ax_radio = fig.add_axes([0.82, 0.55, 0.15, 0.30])
        ax_radio.set_title("plot", fontsize=10)
        radio = RadioButtons(
            ax_radio, mode_options, active=mode_options.index(initial_mode)
        )

    state_holder = {"mode": initial_mode}

    def current_alphas():
        return np.array(
            [sliders[f"alpha_{k}"].val for k in range(num_obstacles)],
            dtype=np.float32,
        )

    def current_forwards():
        return np.array(checks.get_status(), dtype=bool)

    def redraw():
        v = float(sliders["v"].val)
        theta = float(sliders["theta"].val)
        alphas = current_alphas()
        forwards = current_forwards()
        mode = state_holder["mode"]

        grid = np.asarray(
            grid_fns[mode](
                jnp.array(v),
                jnp.array(theta),
                jnp.asarray(alphas),
                jnp.asarray(forwards),
            )
        )

        mesh.set_array(grid.ravel())

        obs_pos = np.asarray(obstacle_positions(jnp.asarray(alphas)))
        for k in range(num_obstacles):
            obstacle_patches[k].center = (obs_pos[k, 0], obs_pos[k, 1])

        title.set_text(f"mode={mode}   v={v:+.2f}   theta={theta:+.2f}")
        fig.canvas.draw_idle()

    def on_release(event):
        if event.inaxes in slider_axes:
            redraw()

    fig.canvas.mpl_connect("button_release_event", on_release)

    checks.on_clicked(lambda _label: redraw())
    if radio is not None:

        def on_radio(label):
            state_holder["mode"] = label
            redraw()

        radio.on_clicked(on_radio)

    plt.show()


if __name__ == "__main__":
    main()
