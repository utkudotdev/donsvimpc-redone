from dynamics.environment_dynamics import State, Parameters
from dynamics.obstacle_dynamics import ObstacleState
from dynamics.dubins_dynamics import DubinsState
import jax
import jax.numpy as jnp
import argparse
from datetime import datetime
from pathlib import Path
from environments.dubins import get_environment_parameters

from functools import partial
import tqdm

from networks.ncbf import NCBF, compute_ncbf_loss
import optax
import equinox as eqx


def make_relative_state(s: State, p: Parameters) -> jnp.ndarray:
    """
    Input to neural-CBF contains the state. To make the neural-CBF more
    general, we pass in a 'relative state'.

    """
    obstacle_relative_pos = (
        jax.vmap(ObstacleState.position)(s.obstacle_state, p.obstacle_params)
        - s.dubins_state.position()
    )
    boundary_max_relative_pos = (
        jnp.array([p.x_max, p.y_max]) - s.dubins_state.position()
    )
    boundary_min_relative_pos = s.dubins_state.position() - jnp.array(
        [p.x_min, p.y_min]
    )

    obstacle_abs_vel = jax.vmap(ObstacleState.velocity)(
        s.obstacle_state, p.obstacle_params
    )

    # TODO: if we randomize car dynamics we would have to include that here
    # Right now, dynamics and velocity constraints are baked into NCBF
    return jnp.concatenate(
        [
            obstacle_relative_pos.flatten(),
            boundary_max_relative_pos,
            boundary_min_relative_pos,
            jnp.atleast_1d(s.dubins_state.v),
            jnp.atleast_1d(s.dubins_state.theta),
            obstacle_abs_vel.flatten(),
        ]
    )


def make_training_triples(relative_states: jnp.ndarray, hs: jnp.ndarray):
    assert relative_states.shape[0] == hs.shape[0]
    return relative_states[:-1], hs[:-1], relative_states[1:]


def split_data(
    all_x_t: jnp.ndarray,
    all_h_t: jnp.ndarray,
    all_x_t1: jnp.ndarray,
    train_percent: float,
    key: jnp.ndarray,
):
    num_triples = all_x_t.shape[0]
    perm = jax.random.permutation(key, all_x_t.shape[0])
    train_count = round(num_triples * train_percent)
    train_idx, test_idx = perm[:train_count], perm[train_count:]
    return (all_x_t[train_idx], all_h_t[train_idx], all_x_t1[train_idx]), (
        all_x_t[test_idx],
        all_h_t[test_idx],
        all_x_t1[test_idx],
    )


@eqx.filter_jit
def train(model, opt_state, optimizer, discount_factor, batch_size, x_t, h_t, x_t1, key):
    """Train for one epoch over shuffled batches. Returns (model, opt_state, cumulative_loss)."""
    n = x_t.shape[0]
    perm = jax.random.permutation(key, n)
    x_t, h_t, x_t1 = x_t[perm], h_t[perm], x_t1[perm]

    num_batches = n // batch_size
    trim = num_batches * batch_size
    x_t = x_t[:trim].reshape(num_batches, batch_size, -1)
    h_t = h_t[:trim].reshape(num_batches, batch_size, -1)
    x_t1 = x_t1[:trim].reshape(num_batches, batch_size, -1)

    # Partition model: only arrays go into the scan carry; static parts
    # (e.g. jax.nn.relu) are captured in the closure.
    dynamic_model, static_model = eqx.partition(model, eqx.is_array)

    def step(carry, batch):
        dynamic_model, opt_state, total_loss = carry
        model = eqx.combine(dynamic_model, static_model)
        x_b, h_b, x_b1 = batch

        def mean_loss(model):
            per_sample = jax.vmap(compute_ncbf_loss, in_axes=(None, None, 0, 0, 0))(
                model, discount_factor, x_b, h_b, x_b1
            )
            return jnp.mean(per_sample)

        loss, grads = eqx.filter_value_and_grad(mean_loss)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        dynamic_model, _ = eqx.partition(model, eqx.is_array)
        return (dynamic_model, opt_state, total_loss + loss), None

    (dynamic_model, opt_state, total_loss), _ = jax.lax.scan(
        step, (dynamic_model, opt_state, jnp.array(0.0)), (x_t, h_t, x_t1)
    )
    model = eqx.combine(dynamic_model, static_model)
    return model, opt_state, total_loss


def save_checkpoint(path: Path, model: NCBF, opt_state, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        eqx.tree_serialise_leaves(f, (model, opt_state, epoch))


def load_checkpoint(path: Path, model_template: NCBF, opt_state_template):
    with open(path, "rb") as f:
        return eqx.tree_deserialise_leaves(f, (model_template, opt_state_template, 0))


@eqx.filter_jit
def evaluate(model, discount_factor, batch_size, x_t, h_t, x_t1):
    """Evaluate over the full dataset. Returns cumulative loss."""
    n = x_t.shape[0]
    num_batches = n // batch_size
    trim = num_batches * batch_size
    x_t = x_t[:trim].reshape(num_batches, batch_size, -1)
    h_t = h_t[:trim].reshape(num_batches, batch_size, -1)
    x_t1 = x_t1[:trim].reshape(num_batches, batch_size, -1)

    def step(total_loss, batch):
        x_b, h_b, x_b1 = batch
        per_sample = jax.vmap(compute_ncbf_loss, in_axes=(None, None, 0, 0, 0))(
            model, discount_factor, x_b, h_b, x_b1
        )
        return total_loss + jnp.mean(per_sample), None

    total_loss, _ = jax.lax.scan(step, jnp.array(0.0), (x_t, h_t, x_t1))
    return total_loss


@eqx.filter_jit
def eval_ncbf_grid(
    ncbf: NCBF, params: Parameters, flat_x: jnp.ndarray, flat_y: jnp.ndarray
):
    num_obstacles = params.obstacle_params.start_point.shape[0]
    obstacle_alpha = jnp.zeros((num_obstacles,))
    obstacle_forward = jnp.ones((num_obstacles,), dtype=bool)

    def _eval_single(x, y):
        state = State(
            dubins_state=DubinsState(x=x, y=y, v=jnp.array(0.1), theta=jnp.array(0.0)),
            obstacle_state=ObstacleState(
                alpha=obstacle_alpha, forward=obstacle_forward
            ),
        )
        rel = make_relative_state(state, params)
        return jnp.max(ncbf(rel))

    return jax.vmap(_eval_single)(flat_x, flat_y)


def visualize_ncbf(
    ncbf: NCBF,
    params: Parameters,
    resolution: int = 100,
    save_path: str | None = None,
    show: bool = True,
):
    """Visualise the trained NCBF over the workspace.

    For every (x, y) grid cell the car is placed there facing +x with a small
    forward velocity while obstacles sit at their start positions with zero
    speed.  The NCBF is evaluated and the *max* of the output h-vector is
    plotted as a heatmap.  Obstacle positions are overlaid as green circles.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Circle

    x_min, x_max = float(params.x_min), float(params.x_max)
    y_min, y_max = float(params.y_min), float(params.y_max)

    xs = jnp.linspace(x_min, x_max, resolution)
    ys = jnp.linspace(y_min, y_max, resolution)
    xx, yy = jnp.meshgrid(xs, ys)  # each (resolution, resolution)

    flat_x = xx.ravel()
    flat_y = yy.ravel()

    num_obstacles = params.obstacle_params.start_point.shape[0]

    max_h = eval_ncbf_grid(ncbf, params, flat_x, flat_y)
    max_h_grid = max_h.reshape(resolution, resolution)

    # ----- Plotting -----
    fig, ax = plt.subplots(figsize=(8, 6))
    vmin = float(jnp.min(max_h_grid))
    vmax = float(jnp.max(max_h_grid))
    # Pad so TwoSlopeNorm is valid even when all values are on one side of 0.
    if vmin >= 0:
        vmin = -1e-6
    if vmax <= 0:
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.pcolormesh(
        xs,
        ys,
        max_h_grid,
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
    )
    fig.colorbar(im, ax=ax, label="max h (NCBF)")

    # Draw obstacles at their start positions
    for i in range(num_obstacles):
        cx, cy = (
            float(params.obstacle_params.start_point[i, 0]),
            float(params.obstacle_params.start_point[i, 1]),
        )
        r = float(params.obstacle_params.radius[i])
        circle = Circle(
            (cx, cy), r, linewidth=2, edgecolor="green", facecolor="green", alpha=0.35
        )
        ax.add_patch(circle)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("NCBF max h(x, y)  — obstacles in green")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        tqdm.tqdm.write(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an NCBF.")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the .npz dataset produced by data collection.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from. If omitted, starts fresh.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data = jnp.load(args.dataset, allow_pickle=True)
    states: State = data["states"].item()
    hs: jnp.ndarray = data["hs"]

    params = get_environment_parameters("basic")

    num_trajs, traj_length = states.dubins_state.x.shape
    h_vec_shape = hs.shape[2]

    @partial(jax.jit, static_argnums=(0,))
    def prepare_dset(train_percent: float, key: jnp.ndarray):
        # input shape: (num_trajs, traj length, state obj ). output shape: (num trajs, traj length, relative state size)
        relative_states = jax.vmap(
            jax.vmap(make_relative_state, in_axes=(0, None)), in_axes=(0, None)
        )(states, params)
        assert relative_states.ndim == 3
        assert relative_states.shape[:2] == (num_trajs, traj_length)
        rel_state_size = relative_states.shape[2]

        x_t, h_t, x_t1 = jax.vmap(make_training_triples)(relative_states, hs)
        assert x_t.shape == (num_trajs, traj_length - 1, rel_state_size), (
            f"x_t.shape was {x_t.shape}"
        )
        assert h_t.shape == (num_trajs, traj_length - 1, h_vec_shape), (
            f"h_t.shape was {h_t.shape}"
        )
        assert x_t1.shape == (num_trajs, traj_length - 1, rel_state_size), (
            f"x_t1.shape was {x_t1.shape}"
        )

        # eliminate batch and traj len dimensions
        all_x_t = x_t.reshape(-1, *x_t.shape[2:])
        all_h_t = h_t.reshape(-1, *h_t.shape[2:])
        all_x_t1 = x_t1.reshape(-1, *x_t1.shape[2:])

        return split_data(all_x_t, all_h_t, all_x_t1, train_percent, key)

    key = jax.random.key(seed=0)

    print("Preparing dataset")

    key, split_data_key = jax.random.split(key)
    train_percent = 0.8
    (train_x_ts, train_h_ts, train_x_t1s), (test_x_ts, test_h_ts, test_x_t1s) = (
        prepare_dset(train_percent, split_data_key)
    )

    epochs = 10
    batch_size = 1024
    discount_factor = 0.92

    run_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    checkpoint_path = checkpoint_dir / "ncbf.eqx"
    print(f"Run directory: {run_dir}")


    rel_state_size = train_x_ts.shape[1]

    key, init_model_key = jax.random.split(key)

    model = NCBF(
        key=init_model_key,
        relative_state_dim=rel_state_size,
        h_vector_dim=h_vec_shape,
        hidden_size=256,
    )

    optimizer = optax.adamw(learning_rate=3e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    start_epoch = 0
    if args.checkpoint is not None:
        print(f"Loading checkpoint from {args.checkpoint}")
        model, opt_state, start_epoch = load_checkpoint(
            args.checkpoint, model, opt_state
        )
        print(f"Resuming from epoch {start_epoch}")


    visualize_ncbf(
        model,
        params,
        save_path=str(plots_dir / f"ncbf_epoch_{start_epoch:03d}.png"),
        show=False,
    )

    for epoch in tqdm.trange(start_epoch, epochs, initial=start_epoch, total=epochs):
        key, epoch_key = jax.random.split(key)
        model, opt_state, train_loss = train(
            model, opt_state, optimizer, discount_factor, batch_size,
            train_x_ts, train_h_ts, train_x_t1s, epoch_key,
        )
        test_loss = evaluate(
            model, discount_factor, batch_size,
            test_x_ts, test_h_ts, test_x_t1s,
        )

        visualize_ncbf(
            model,
            params,
            save_path=str(plots_dir / f"ncbf_epoch_{epoch + 1:03d}.png"),
            show=False,
        )

        save_checkpoint(checkpoint_path, model, opt_state, epoch + 1)

        tqdm.tqdm.write(
            f"Epoch {epoch + 1} | Train Loss: {float(train_loss):.4f} | Test Loss: {float(test_loss):.4f}"
        )


if __name__ == "__main__":
    main()
