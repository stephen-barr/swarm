# Consensus-Based Decentralized Auctions for Heterogeneous Drone Swarms
[![tests](https://github.com/stephen-barr/swarm/actions/workflows/tests.yml/badge.svg)](https://github.com/stephen-barr/swarm/actions/workflows/tests.yml)

This is a from-scratch Python implementation of decentralized task allocation for a heterogeneous drone swarm. The fundamental problem it aims to solve is as follows:

***How can a decentralized swarm agree on coalition assignments when task demand, communication and the task set itself are all uncertain?***

Such a question is of broad interest to the field of autonomous systems. Consider the example below to illustrate its practicality:

*Medical triage via unmanned vehicles requires real-time coalition assignments with uncertain priority and capability requirements. Such systems are often needed in hostile environments where RF communication is jammed, leading to packet loss. How can a swarm accurately allocate capabilities to casualties?*

## Status and Direction

Currently this project implements decentralized single-assignment auctions (CBAA) with multi-capability roles, checked against the optimal assignment found through a Hungarian algorithm.

I aim to incrementally add uncertainty to a working implementation of the standard consensus auction algorithm.

1. **Single assignment (CBAA): done.** Everything is known and fixed: tasks, demand, and a reliable, synchronous network.
2. **Bundles (CBBA): next.** Not yet introducing uncertainty. Agents are assigned ordered lists of targets instead of a single assignment. Coalitions in the CBBA line are built on bundles, so this comes first.
3. **Coalitions under uncertain demand: partial.** First uncertainty: how many drones a target actually needs. Multi-capability roles and multi-drone targets are implemented already, but slots are auctioned independently (no coupled constraints) and demand is fixed at its most likely value before bidding starts.
4. **Degraded communication: planned.** Second uncertainty: message loss between agents. Implemented to measure the robustness of the algorithm to disruptions. The channel supports a fixed delay, currently.
5. **Tasks arriving mid-mission: planned.** Third uncertainty: which tasks exist. Solves real-time target identification from perception, with the cost of changing a drone's current assignment.

## Assumptions (current)
 
- Synchronous rounds over a static, connected, symmetric communication graph.
- One slot per drone.
- Drones and targets are stationary; positions are in the global frame.
- Demand is fixed at its most likely count before bidding.


## Scoring: time-discounted value

Each drone scores a slot by the target's value, discounted by how long the drone would take to get there:

$$\text{score} = V \cdot e^{-t/\tau}$$

- **V** is the target's value to the mission (currently `threat_level`).
- **t** is the drone's own time to arrival (distance ÷ cruise speed), so scores are local to each drone.
- **τ** (tau) is a characteristic time: value halves every 0.69 τ seconds, so τ = 30 s means value halves roughly every 21 s.

**Why this form**


- **Scores are never negative**, so any assignment the auction makes is worth something.

- **It supports bundles.** Adding tasks to a drone's path can only delay the others, so marginal gains only decrease, the condition CBBA's convergence and optimality guarantees require.

**Behaviour**

- V sets how high a target's curve starts; τ sets how quickly it falls.
- Curves can cross: an urgent, valuable target (V = 0.8, τ = 30 s) becomes worth less than a patient, modest one (V = 0.3, τ = 90 s) after about 44 s of travel. Nearby drones prefer the first; distant ones, the second.

<p align="center">
<img width="610" alt="discounted_value" src="https://github.com/user-attachments/assets/c38e5a39-c8dc-48a5-a211-bc649c20e1c6" />
</p>

## Tests
 
```bash
pip install -e .
pytest
```
 
- `tests/test_comm.py`: channel delivery, copying, ordering, delay.
- `tests/test_auction.py`: hand-built cases, plus properties over random instances on complete, ring and line graphs (conflict-free, capabilities respected, terminates, tables agree, deterministic, within ½·OPT ≤ auction ≤ OPT).
- `tests/test_baseline.py`: Hungarian baseline against brute force.
## Layout
 
```
swarm/
  models.py      shared types
  agent.py       scoring, bidding, merging
  comms.py       simulated network
  auction.py     rounds and stopping rule
  baselines.py   Hungarian baseline
  scenarios.py   graph builders, fixtures, random instances
experiments/
  basic_example.py
tests/
```
 
## References
 
- Choi, H.-L., Brunet, L., How, J. P. Consensus-Based Decentralized Auctions for Robust Task Allocation. *IEEE Transactions on Robotics* 25(4), 2009.
- Hunt, S., Meng, Q., Hinde, C., Huang, T. A Consensus-Based Grouping Algorithm for Multi-agent Cooperative Task Allocation with Complex Requirements. *Cognitive Computation* 6, 2014.
- Kuhn, H. W. The Hungarian Method for the Assignment Problem. *Naval Research Logistics Quarterly* 2, 1955.



All rights are reserved.
