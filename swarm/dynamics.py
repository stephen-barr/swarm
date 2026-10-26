from dataclasses import replace

import numpy as np

from swarm.models import AgentState, PlatformSpec, ThreeVector

_g = 9.81


def advance(spec: PlatformSpec, state: AgentState, accel_command: ThreeVector,
            dt: float) -> AgentState:
    x = np.array(state.estimate.position,
                 dtype=float)               ## Z does not change in current implementations
    v = np.array(state.estimate.velocity, dtype=float)
    a = np.array(accel_command, dtype=float)

    v[2] = 0.0          # do not consider changes in altitude
    a[2] = 0.0

    if not spec.can_hover:
        """
        Fixed wing, find lateral and parallel components of acceleration command to heading.
        Clip everything to within maximum and minimum values.
        Compute new values and return
        """
        # signed scalar component of acceleration along velocity
        speed = np.linalg.norm(v)
        if speed == 0:
            raise ValueError("Speed cannot be zero for fixed wing."
                             )                                      # ensure speed is non-zero
        v_hat = v / speed
        a_long = np.clip(
            a @ v_hat, spec.a_min,
            spec.a_max)                                             # clip acceleration based on forward/backward limits
        new_speed = np.clip(speed + a_long * dt, spec.v_min, spec.v_max)

        # acceleration perpendicular to velocity
        n_hat = np.array(
            [-v_hat[1], v_hat[0], 0]
        )                                                                       # normalized unit vector orthogonal to v (counter-clockwise convention)
        a_lat_no_clip = a @ n_hat
        phi_cmd = np.clip(np.arctan(a_lat_no_clip / _g),
                          -np.radians(spec.max_bank), np.radians(spec.max_bank)
                          )                                                     # clip the bank change to maximum and minimum bank

        max_roll_step = dt * np.radians(
            spec.max_roll_rate
        )                                             # get the maximum roll possible in this step in rad/s
        new_bank = state.bank + np.clip(
            phi_cmd - state.bank, -max_roll_step,
            max_roll_step)                            # roll changes clipped to max_roll_step

        a_lat = _g * np.tan(
            new_bank)             # return the lateral acceleration based on the roll

        # find the new heading
        phi_before = np.arctan2(
            v[1], v[0]
        )                                                         # arctan to find current heading, returns back correct branch (counter-clockwise convention)
        omega = (a_lat / new_speed)                               # find rotational velocity
        new_phi = phi_before + omega * dt                         # new heading
        new_v = new_speed * np.array([np.cos(new_phi),
                                      np.sin(new_phi), 0.0])      # new velocity
        new_x = x + new_v * dt                                    # new position
        pos_estimate = replace(state.estimate,
                               position=tuple(map(float, new_x)),
                               velocity=tuple(map(float, new_v)))
        new_state = replace(state, estimate=pos_estimate, bank=new_bank)

        return new_state

    else:
        """Multirotor, no need to orthogonalize acceleration."""
        a_clip = clipped_norm(a, spec.max_lateral_accel
                              )                                   # all acceleration for multiorot comes from tilt
        new_v = clipped_norm(a_clip * dt + v, spec.v_max)
        new_x = x + new_v * dt
        pos_estimate = replace(state.estimate,
                               position=tuple(map(float, new_x)),
                               velocity=tuple(map(float, new_v)))
        new_state = replace(state, estimate=pos_estimate)

        return new_state


def clipped_norm(
    vec: np.ndarray, limit: float
) -> np.ndarray:                            # helper function returns clipped magnitude, same direction
    norm = np.linalg.norm(vec)
    if norm > limit:
        return vec * limit / norm
    return vec
