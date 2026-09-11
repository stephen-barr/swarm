
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from typing import Any, Callable, Hashable, Iterable, Optional
import unittest

@dataclass
class graph:
    n: int
    r: float
    seed: int
    edges: list[list[float]] = field(default_factory=list)
    vertices: np.ndarray = None

    def __post_init__(self):
        if self.vertices is None:
            #construct the vertices with association given by the seed
            self.vertices = np.random.default_rng(self.seed).random((self.n,3))
            self.vertices[:,2] = 0.0
        if not self.edges:
            tree = cKDTree(self.vertices)
            #return sorted 
            self.edges = sorted(tree.query_pairs(self.r))
        

@dataclass
class Message:
    sender: int
    receiver: int
    payload: Any
    send_step: int
    deliver_step: int
    seq: int

    def edge(self) -> Edge:
        return(self.sender, self.receiver)
    
    def age(self, t: int) -> int:
        return t - self.send_step


class Channel:
    def __init__(self,
        n_agents: int,
        seed: int,
        directed_graph: bool = False,
        ):
        self.n = n_agents
        self.seed = seed
        self.directed_graph = directed_graph
        self._t = -1
        self.up: set = set()
        self.log: list = []
        self._intended: set[Edge] = set()
        self._inbox: dict[int, list[Messages]] = {}
        self._seq: dict[int, int] = {i: 0 for i in range(n_agents)}

    def begin_step(self, t: int, edges) -> None:
        assert t == self._t + 1, f"Error: steps must by one, at {self._t} got {t} "
        self._t = t
        self._intended = set()
        self._inbox = {i: [] for i in range(self.n)}
        for a, b in edges:
            assert a != b, "Error: Trying to send message to self"
            self._intended.add((a,b))
            # if undirected graph add other pair to intended
            if not self.directed_graph:
                self._intended.add((b,a))

    def broadcast(self, sender: int, payload: Any) -> None:
        seq = self._seq[sender]
        self._seq[sender] += 1
        for i,j in self._intended:
            if i != sender:
                continue
            self._inbox[j].append(Message(sender, j, payload, self._t, self._t, seq))    
    def deliver(self, receiver: int) -> list[Message]:
        msgs = self._inbox[receiver]
        self._inbox[receiver] = []
        msgs.sort(key = lambda m: (m.sender, m.seq))
        return msgs
    def end_step(self) -> None:
        leftover = {i: len(v) for i, v in self._inbox.items() if v}
        assert not leftover, f"undelivered at step {self._t}: {leftover}"

def update_heading(theta_i: float, payloads: list[float], eps: float):
    z = np.exp(1j*theta_i)
    for p in payloads:
        z += eps*(np.exp(1j*p)-np.exp(1j*theta_i))
    return float(np.angle(z))

def laplacian(n, edges):
    L = np.zeros((n,n))
    for i, j in edges:
        L[i, j] += -1
        L[j, i] += -1
        L[i, i] += 1
        L[j, j] += 1
    return L

def global_consistency(theta):
    return abs(np.mean(np.exp(1j * theta)))

def run(g, theta0, eps, T, seed=0):
    """ Returns (T, n) heading history. 
    
    Utilizes Jacobi method to update all agents
    from the pre-step state.

    No silent-neighbour policy. An agent averages only over what
    arrived this step.
    """

    ch = Channel(g.n, seed, directed_graph=False)
    theta = theta0.copy()
    theta_hist = []
    for t in range(T):
        ch.begin_step(t, g.edges)
        for i in range(g.n):            # broadcast the current payload to all connected neighbors
            ch.broadcast(i, theta[i]) 
        theta_next = theta.copy()       
        for i in range(g.n):
            theta_next[i] = update_heading(theta[i], [m.payload for m in ch.deliver(i)], eps)
        ch.end_step()
        theta = theta_next              # theta changes only after all messages are recieved
        theta_hist.append(theta.copy())
    return np.array(theta_hist)
 
if __name__ == "__main__":
    g = graph(n=12, r=0.45, seed=1)
    lam = np.sort(np.linalg.eigvalsh(laplacian(g.n, g.edges)))
    lam2, lamN = lam[1], lam[-1]
    bound = 2.0 / lamN
    theta0 = np.random.default_rng(7).normal(0, 0.25, g.n)      #random heading values

    print(f"lambda2={lam2:.4f}  lambdaN={lamN:.4f}  stability bound eps<{bound:.4f}\n")
    print(f"{'eps':>8} {'eps/bound':>10} {'final disag':>14} {'pred rate':>11} {'meas rate':>11}")

    for frac in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0, 1.01, 1.1, 1.3]:
        eps = frac * bound
        hist = run(g, theta0, eps, 120)
        d = np.linalg.norm(hist - hist.mean(1, keepdims=True), axis=1)
        pred = np.log(max(abs(1 - eps*lam2), abs(1 - eps*lamN)))
        if np.all(np.isfinite(d)) and d[-1] > 0:
            meas = np.polyfit(range(60, 110), np.log(d[60:110]), 1)[0]
            meas_s, fin = f"{meas:11.5f}", f"{d[-1]:14.3e}"
        else:
            meas_s, fin = f"{'diverged':>11}", f"{'diverged':>14}"
        print(f"{eps:8.4f} {frac:10.2f} {fin} {pred:11.5f} {meas_s}")

    
