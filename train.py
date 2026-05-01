from dynamics.environment_dynamics import State, Parameters
from dynamics.obstacle_dynamics import ObstacleState
import jax
import jax.numpy as jnp
import sys
from environments.dubins import get_environment_parameters
import numpy as np
from functools import partial

from networks.ncbf import NCBF, compute_ncbf_loss
import optax
import equinox as eqx

def make_relative_state(s: State, p: Parameters) -> jnp.ndarray:
    """ 
    Input to neural-CBF contains the state. To make the neural-CBF more
    general, we pass in a 'relative state'. 
      
    """
    obstacle_relative_pos = jax.vmap(ObstacleState.position)(s.obstacle_state, p.obstacle_params) - s.dubins_state.position()
    boundary_max_relative_pos = jnp.array([p.x_max, p.y_max]) - s.dubins_state.position()
    boundary_min_relative_pos = s.dubins_state.position() - jnp.array([p.x_min, p.y_min])
    
    obstacle_abs_vel = jax.vmap(ObstacleState.velocity)(s.obstacle_state, p.obstacle_params)

    # TODO: if we randomize car dynamics we would have to include that here
    # Right now, dynamics and velocity constraints are baked into NCBF
    return jnp.concatenate([
        obstacle_relative_pos.flatten(),
        boundary_max_relative_pos,
        boundary_min_relative_pos,
        jnp.atleast_1d(s.dubins_state.v),
        jnp.atleast_1d(s.dubins_state.theta),
        obstacle_abs_vel.flatten()
    ])



def make_training_triples(relative_states: jnp.ndarray, hs: jnp.ndarray) -> jnp.ndarray:
    assert relative_states.shape[0] == hs.shape[0]
    return relative_states[:-1], hs[:-1], relative_states[1:]


def split_data(all_x_t: jnp.ndarray, all_h_t: jnp.ndarray, all_x_t1: jnp.ndarray, train_percent: float, key: jnp.ndarray) -> jnp.ndarray:
    num_triples = all_x_t.shape[0]
    perm = jax.random.permutation(key, all_x_t.shape[0])
    train_count = round(num_triples * train_percent)
    train_idx, test_idx = perm[:train_count], perm[train_count:]
    return (all_x_t[train_idx], all_h_t[train_idx], all_x_t1[train_idx]), (all_x_t[test_idx], all_h_t[test_idx], all_x_t1[test_idx])


def dataloader(arrays, batch_size):
    flat_arrays = [arr.reshape(-1, arr.shape[-1]) for arr in arrays]
    
    dataset_size = flat_arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in flat_arrays)
    indices = np.arange(dataset_size)
    
    def _impl():
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in flat_arrays)
            start = end
            end = start + batch_size
            
    return _impl

def visualize_ncbf(ncbf: NCBF, params: Parameters, resolution: int = 100, save: bool = False):
    """Visualise the trained NCBF over the workspace.

    For every (x, y) grid cell the car is placed there facing +x with a small
    forward velocity while obstacles sit at their start positions with zero
    speed.  The NCBF is evaluated and the *max* of the output h-vector is
    plotted as a heatmap.  Obstacle positions are overlaid as green circles.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    from dynamics.dubins_dynamics import DubinsState
    from dynamics.obstacle_dynamics import ObstacleState

    x_min, x_max = float(params.x_min), float(params.x_max)
    y_min, y_max = float(params.y_min), float(params.y_max)

    xs = jnp.linspace(x_min, x_max, resolution)
    ys = jnp.linspace(y_min, y_max, resolution)
    xx, yy = jnp.meshgrid(xs, ys)  # each (resolution, resolution)

    # Flatten grid to (resolution^2,)
    flat_x = xx.ravel()
    flat_y = yy.ravel()

    num_obstacles = params.obstacle_params.start_point.shape[0]

    # Build a batched State at every grid point.
    # Obstacles: stationary at their start positions (alpha=0, forward=True).
    obstacle_alpha = jnp.zeros((num_obstacles,))
    obstacle_forward = jnp.ones((num_obstacles,), dtype=bool)

    def _eval_single(x, y):
        state = State(
            dubins_state=DubinsState(x=x, y=y, v=jnp.array(0.1), theta=jnp.array(0.0)),
            obstacle_state=ObstacleState(alpha=obstacle_alpha, forward=obstacle_forward),
        )
        rel = make_relative_state(state, params)
        h_vec = ncbf(rel)
        return jnp.max(h_vec)

    max_h = jax.vmap(_eval_single)(flat_x, flat_y)
    max_h_grid = max_h.reshape(resolution, resolution)

    # ----- Plotting -----
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        xs, ys, max_h_grid,
        cmap="RdYlGn_r",
        shading="auto",
    )
    fig.colorbar(im, ax=ax, label="max h (NCBF)")

    # Draw obstacles at their start positions
    for i in range(num_obstacles):
        cx, cy = float(params.obstacle_params.start_point[i, 0]), float(params.obstacle_params.start_point[i, 1])
        r = float(params.obstacle_params.radius[i])
        circle = Circle((cx, cy), r, linewidth=2, edgecolor="green", facecolor="green", alpha=0.35)
        ax.add_patch(circle)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("NCBF max h(x, y)  — obstacles in green")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    if save:
        plt.savefig("ncbf_heatmap.png", dpi=200)
        print("Saved ncbf_heatmap.png")
    plt.show()

def main():
    data = jnp.load(sys.argv[1], allow_pickle=True)
    states: State = data['states'].item()
    hs: jnp.ndarray = data['hs']

    params = get_environment_parameters('basic')
    
    num_trajs, traj_length = states.dubins_state.x.shape
    h_vec_shape = hs.shape[2]


    @partial(jax.jit, static_argnums=(0, ))
    def prepare_dset(train_percent: float, key: jnp.ndarray):
        # input shape: (num_trajs, traj length, state obj ). output shape: (num trajs, traj length, relative state size)
        relative_states = jax.vmap(jax.vmap(make_relative_state, in_axes=(0, None)), in_axes=(0, None))(states, params)
        assert relative_states.ndim == 3
        assert relative_states.shape[:2] == (num_trajs, traj_length)
        rel_state_size = relative_states.shape[2]
        
        x_t, h_t, x_t1 = jax.vmap(make_training_triples)(relative_states, hs)
        assert x_t.shape == (num_trajs, traj_length - 1, rel_state_size), f'x_t.shape was {x_t.shape}'
        assert h_t.shape == (num_trajs, traj_length - 1, h_vec_shape), f'h_t.shape was {h_t.shape}'
        assert x_t1.shape == (num_trajs, traj_length - 1, rel_state_size), f'x_t1.shape was {x_t1.shape}'

        all_x_t = x_t.reshape(-1, *x_t.shape[1:])
        all_h_t = h_t.reshape(-1, *h_t.shape[1:])
        all_x_t1 = x_t1.reshape(-1, *x_t1.shape[1:])

        return split_data(all_x_t, all_h_t, all_x_t1, train_percent, key)
    
    key = jax.random.key(seed=0)

    key, split_data_key = jax.random.split(key)
    train_percent = 0.8
    (train_x_ts, train_h_ts, train_x_t1s), (test_x_ts, test_h_ts, test_h_t1s) = prepare_dset(train_percent, split_data_key)
    #NOTE: add checkpointing and loading from a checkpoint
    
    # TODO:
    #   - evaluate on held-out set of states
    #   - add checkpointing and loading from a checkpoint
    #   - save plots of V_h (instead of displaying them during training)

    epochs = 10
    batch_size = 128
    discount_factor = 0.99

    train_dataloader = dataloader([train_x_ts, train_h_ts, train_x_t1s], batch_size)
    test_dataloader = dataloader([test_x_ts, test_h_ts, test_h_t1s], batch_size)

    rel_state_size = train_x_ts.shape[2]
    

    key, init_model_key = jax.random.split(key)
    
    
    
    model = NCBF(
        key=init_model_key,
        relative_state_dim=rel_state_size, 
        h_vector_dim=h_vec_shape, 
        hidden_size=256)

    optimizer = optax.adamw(learning_rate=3e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def train_step(model, opt_state, x_t_batch, h_t_batch, x_t1_batch):
        def compute_mean_loss(model, discount_factor, x_t_batch, h_t_batch, x_t1_batch):
            batch_loss_fn = jax.vmap(compute_ncbf_loss, in_axes=(None, None, 0, 0, 0))
            batch_loss = batch_loss_fn(model, discount_factor, x_t_batch, h_t_batch, x_t1_batch)
            return jnp.mean(batch_loss, axis=0)

        loss, grads = eqx.filter_value_and_grad(compute_mean_loss)(
            model, discount_factor, x_t_batch, h_t_batch, x_t1_batch)

        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    

    visualize_ncbf(model, params)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for x_t_batch, h_t_batch, x_t1_batch in train_dataloader():
            model, opt_state, loss = eqx.filter_jit(train_step)(model, opt_state, x_t_batch, h_t_batch, x_t1_batch)

            num_batches += 1
            epoch_loss += loss.item()

        visualize_ncbf(model, params)

        print(f"Epoch {epoch + 1} | Epoch Loss: {epoch_loss:.4f}")

if __name__ == '__main__':
    main()
