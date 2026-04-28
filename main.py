from jax.tree_util import register_dataclass

from environment.environment import Parameters, State, step_state
from environment.obstacle_dynamics import ObstacleState, step_obstacle
from environment.quadrotor_dynamics import QuadrotorState, step_quadrotor


def main():
    print("Hello from donsvimpc-redone!")


if __name__ == "__main__":
    main()
