import numpy as np

from swarm.dynamics import advance
from swarm.guidance import goToTarget, loiterCommand
from swarm.models import (
    AgentState,
    Capability,
    Frame,
    MissionData,
    PlatformClass,
    PositionEstimate,
    role,
)
from swarm.scenarios import at, make_agent, make_target

S = Capability.SURVEILLANCE
M = Capability.MAPPING
spec_multirotor = PlatformClass.GROUP_1_MULTIROTOR.value
spec_fixed_wing = PlatformClass.GROUP_1_FIXED_WING.value
state = AgentState(estimate=at(0, 0, 0))
state_fixed_wing = AgentState(
    estimate=PositionEstimate(frame=Frame.GLOBAL,
                              position=(-300.0, 0.0, 0.0),
                              velocity=(0, spec_fixed_wing.v_cruise, 0.0),
                              position_covariance=((1, 0, 0), (0, 1, 0), (0, 0,
                                                                          1))))
target = make_target(
    target_id=0,
    position=at(200, 200, 0),
    threat=0.9,
    requirements={
        role(S): {
            1: 1.0
        },                        # one SURVEILLANCE drone, certain
        role(S, M): {
            1: 0.3,
            2: 0.7
        },                        # one drone with both SURVEILLANCE and MAPPING; 70% chance it needs two
    })


def fly_multirotor(spec, state, target_pos, dt, t_end):
    """
    Test only. Multirotor
    Run gotoTarget + advance in closed loop; return the list of states.
    """
    states = [state]
    for _ in range(int(t_end / dt)):
        cmd = goToTarget(spec, state, target_pos)
        state = advance(spec, state, cmd, dt)
        states.append(state)
    return states


def fly_fixed_wing(spec, state, target_pos, dt, t_end):
    states = [state]
    for _ in range(int(t_end / dt)):
        cmd = loiterCommand(spec, state, target_pos, 1)
        state = advance(spec, state, cmd, dt)
        states.append(state)
    return states


def test_multirotor_start_go_stop():
    states = fly_multirotor(spec_multirotor, state,
                            target.target_position.position, 0.1, 100)
    dt = 0.1
    goal = np.array(target.target_position.position)
    dists = [
        np.linalg.norm(np.array(s.estimate.position) - goal) for s in states
    ]
    arrival_step = next(i for i, d in enumerate(dists) if d < 1.0)
    print(f"arrived after {arrival_step * dt:.1f} s")
    start = np.array(state.estimate.position)
    u = (goal - start) / np.linalg.norm(goal - start)
    progress = [(np.array(s.estimate.position) - start) @ u for s in states]

    assert np.linalg.norm(
        np.array(states[-1].estimate.position) -
        np.array(target.target_position.position)
    ) <= 1e-1                                          # position is within tolerance
    assert np.linalg.norm(states[-1].estimate.velocity
                          ) < 0.5                      # final velocity at target is within tolerance

    assert max(progress) <= np.linalg.norm(goal -
                                           start) + 1.0     # never overshot target
    assert max(
        np.linalg.norm(s.estimate.velocity) for s in states
    ) <= spec_multirotor.v_cruise + 0.1                     # never exceeded cruise velocity


def test_fixed_wing_orbit():
    states = fly_fixed_wing(spec=spec_fixed_wing,
                            state=state_fixed_wing,
                            target_pos=target.target_position.position,
                            dt=0.1,
                            t_end=100)
    last_states = states[-300:]                                  # only last 30 seconds of flight
    goal = np.array(target.target_position.position)[:2]
    dists = [
        np.linalg.norm(np.array(s.estimate.position[:2]) - goal)
        for s in states
    ]
    last_dists = dists[-300:]                                    # only look at the last 30 seconds (450 frames)

    speeds = [
        np.linalg.norm(np.array(s.estimate.velocity[:2])) for s in states
    ]
    assert all(
        abs(d - spec_fixed_wing.loiter_radius) <= 0.05 *
        spec_fixed_wing.loiter_radius for d in last_dists)

    for s in last_states:
        rx, ry = np.array(
            s.estimate.position[:2]) - goal           # position relative to the target
        vx, vy = s.estimate.velocity[:2]
        assert rx * vy - ry * vx > 0                  # assert rotation is counterclockwise based on cross product of r x v

    assert all(
        abs(
            np.linalg.norm(np.array(s.estimate.velocity)) -
            spec_fixed_wing.loiter_speed) <= 0.1 * spec_fixed_wing.loiter_speed
        for s in
        last_states)                                                            # assert the speed is approximately the loiter speed
    assert all(
        abs(s.bank) <= np.radians(spec_fixed_wing.max_bank) + 1e-9
        for s in states)                                                        # assert that the maximum bank is never exceeded

    assert all(
        abs(s.bank) < np.radians(spec_fixed_wing.max_bank) - np.radians(5)
        for s in last_states)                                              # assert the bank angle has tolerance in orbit

    assert all(speed <= spec_fixed_wing.v_cruise + 0.5 for speed in
               speeds)                                              # assert the speed is never greater than the max speed


def test_step():
    fixed_wing_agent = make_agent(
        agent_id=0,                                    # make two agents
        capabilities=[S, M],
        x=-300,
        y=-200,
        platform=PlatformClass.GROUP_1_FIXED_WING)
    multirotor_agent = make_agent(agent_id=0,
                                  capabilities=[S, M],
                                  x=-300,
                                  y=-200,
                                  platform=PlatformClass.GROUP_1_MULTIROTOR)

    mission_data = MissionData.fromTargets([target])
    dt = 0.1
    for t in range(2000):         # let it run for 200 seconds
        fixed_wing_agent.step(mission_data=mission_data, dt=dt)
        multirotor_agent.step(mission_data=mission_data, dt=dt)

    final_fixed_agent_position = np.array(
        fixed_wing_agent.state.estimate.position)
    final_mr_agent_position = np.array(
        multirotor_agent.state.estimate.position)
    target_position = np.array(target.target_position.position)
    loiter_radius = fixed_wing_agent.config.platform.value.loiter_radius

    assert np.linalg.norm(
        final_fixed_agent_position - target_position
    ) <= 1.1 * loiter_radius                          # ensure the fixed wing orbits the target relatively close to loiter_radius
    assert np.linalg.norm(
        final_mr_agent_position - target_position
    ) <= 1.0                                          # ensure the multirotor gets within a meter of the target
