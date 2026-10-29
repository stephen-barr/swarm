import numpy as np

from swarm.models import AgentState, PlatformSpec, ThreeVector


def goToTarget(spec: PlatformSpec, state: AgentState,
               target_position: ThreeVector) -> ThreeVector:
    """
    Returns acceleration ThreeVector from agent to target with braking.
    """
    p = np.array(state.estimate.position[:2],
                 dtype=float)                         # get x, y positions of agent, not considering z
    v = np.array(state.estimate.velocity[:2],
                 dtype=float)                         # get velocity, not considering z
    c = np.array(target_position[:2], dtype=float)    # x, y positions of target
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


def loiterCommand(spec: PlatformSpec,
                  state: AgentState,
                  target_position: ThreeVector,
                  direction: int,
                  k: float = 1.0) -> ThreeVector:
    """
    Implementing circular orbit following. 
    Uses +1 counterclockwise as convention.
    Note that kappa is very large when far, 
    so more radial, small when close so tangential.
    """
    if abs(direction) != 1:
        raise ValueError("Direction must be positive or negative one.")
    p = np.array(state.estimate.position[:2],
                 dtype=float)                         # get x, y positions of agent, not considering z
    v = np.array(state.estimate.velocity[:2],
                 dtype=float)                         # get velocity, not considering z
    c = np.array(target_position[:2], dtype=float)    # x, y positions of target
    r = p - c                                         # vector pointing from the target to the agent
    dist = np.linalg.norm(r)                          # distance from agent to target
    if dist < 1e-2:                                   # if within a centimeter, go east
        r_hat = np.array([1.0, 0.0])
    else:
        r_hat = r / dist                              # vector pointing out from target
    t_hat = direction * np.array([-r_hat[1], r_hat[0]
                                  ])                  # vector directly tangent to circle
    R = spec.loiter_radius                            # set a radius for the circle from specs
    kappa = k * (
        dist - R
    ) / R                                             # find the error from the path, positive on outside, negative on inside
    u = t_hat - kappa * r_hat                         # go tangent to the circle, if kappa negative push out, if positive push in
    v_des = spec.loiter_speed * u / np.linalg.norm(u) # renormalize the speed
    v_tang = v @ t_hat                                # get the tangenetial part of velocity
    a_ff = -(
        np.linalg.norm(v_tang)**2 / R
    ) * r_hat                                         # feedforward acceleration needed so agent does not drift tangentially out.
    a_fb = (v_des - v) / spec.tau                     # close the velocity error in char time tau
    a_cmd = a_ff + a_fb
    return (float(a_cmd[0]), float(a_cmd[1]), 0.0)
