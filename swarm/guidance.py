import numpy as np

from swarm.models import AgentState, PlatformSpec, Target, ThreeVector


def gotoTarget(spec: PlatformSpec, state: AgentState,
               target: Target) -> ThreeVector:
    """
    Returns acceleration ThreeVector from agent to target with braking.
    """
    p = np.array(state.estimate.position[:2],
                 dtype=float)                         # get x, y positions of agent, not considering z
    v = np.array(state.estimate.velocity[:2],
                 dtype=float)                         # get velocity, not considering z
    c = np.array(target.target_position.position[:2],
                 dtype=float)                         # x, y positions of target
    delta = c - p                                     # vector pointing from the agent to the target
    dist = np.linalg.norm(delta)                      # distance from agent to target

    if dist < 1e-1:                                       # tolerance within 10 centimeters
        new_vel = np.zeros(2)
    else:
        a_b = 0.8 * abs(
            spec.a_min
        )                                                 # planned braking deceleration (minimum acceleration), ensure positive for sqrt
        braking_speed_fastest = np.sqrt(
            2 * a_b * dist
        )                                                 # kinematically the fastest one could brake at deceleration of 0.8 a_min
        critically_damped_speed = dist / (
            4 * spec.tau
        )                                                 # critically damped velocity, for smooth braking close to target
        speed = min(spec.v_cruise, braking_speed_fastest,
                    critically_damped_speed
                    )                                     # take the minimum of any of the three models
        new_vel = speed * delta / dist                    # velocity vector pointing to target with magnitude set by speed
    a = (
        new_vel - v
    ) / spec.tau                                          # comes from def. of tau, how quickly one corrects an error in velocity.
    a_cmd = (float(a[0]), float(a[1]), 0.0)               # return a ThreeVector
    return a_cmd
