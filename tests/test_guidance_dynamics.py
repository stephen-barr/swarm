import numpy as np

from swarm.dynamics import advance
from swarm.guidance import gotoTarget
from swarm.models import AgentState, Capability, PlatformClass, role
from swarm.scenarios import at, make_target

S = Capability.SURVEILLANCE
M = Capability.MAPPING
spec = PlatformClass.GROUP_1_MULTIROTOR.value
state = AgentState(estimate=at(0, 0, 0))
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


def fly(spec, state, target, dt, t_end):
    """
    Test only.
    Run gotoTarget + advance in closed loop; return the list of states.
    """
    states = [state]
    for _ in range(int(t_end / dt)):
        cmd = gotoTarget(spec, state, target)
        state = advance(spec, state, cmd, dt)
        states.append(state)
    return states


def test_multirotor_start_go_stop():
    states = fly(spec, state, target, 0.1, 100)
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
                                           start) + 1.0 # never overshot target
    assert max(
        np.linalg.norm(s.estimate.velocity)
        for s in states) <= spec.v_cruise + 0.1         # never exceeded cruise velocity
