# Consensus-Based Decentralized Auctions for Heterogenous Drone Swarms
[![tests](https://github.com/stephen-barr/swarm/actions/workflows/tests.yml/badge.svg)](https://github.com/stephen-barr/swarm/actions/workflows/tests.yml)

This is an implementation and extension of the Consensus-Based Grouping Algorithm, an algorithm solving multi-agent task allocation (and thereby an extension of the single-agent tast allocation CBAA) as outlined in Hunt et. al. 2018. 

## Scoring: time-discounted value

Each drone scores a slot by the target's value, discounted by how long the drone would take to get there:

$$\text{score} = V \cdot e^{-t/\tau}$$

- **V** is the target's value to the mission (currently `threat_level × 100`).
- **t** is the drone's own time to arrival (distance ÷ cruise speed), so scores are local to each drone.
- **τ** (tau) is a characteristic time: value halves every 0.69 τ seconds, so τ = 30 s means value halves roughly every 21 s.

**Why this form**

- **Urgency scales with value.** Each second of delay costs the same *fraction* of value, so a high-value target loses more in absolute terms per second than a low-value one.
- **Scores are never negative**, so any assignment the auction makes is worth something.
- **One interpretable parameter.** τ replaces a hidden conversion constant between threat and seconds in the earlier linear score (`threat × 100 − t`).
- **It supports bundles.** Adding tasks to a drone's path can only delay the others, so marginal gains only decrease, the condition CBBA's convergence and optimality guarantees require.

**Behaviour**

- V sets how high a target's curve starts; τ sets how quickly it falls.
- Curves can cross: an urgent, valuable target (V = 80, τ = 30 s) becomes worth less than a patient, modest one (V = 30, τ = 90 s) after about 44 s of travel. Nearby drones prefer the first; distant ones, the second.

<p align="center">
<img width="610" alt="discounted_value" src="https://github.com/user-attachments/assets/c38e5a39-c8dc-48a5-a211-bc649c20e1c6" />
</p>


**Notes**

- The optimal (Hungarian) baseline uses the same score function, so gaps between it and the auction are comparable.
- Results depend on τ, which is recorded with every experiment.
