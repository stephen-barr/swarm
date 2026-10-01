import numpy as np
import scipy


def line_edges(n) -> tuple[int, list[tuple[int, int]]]:
    return n, [(i, i + 1) for i in range(n - 1)]


def complete_edges(n) -> tuple[int, list[tuple[int, int]]]:
    return n, [(i, j) for i in range(n) for j in range(n) if i < j]


def ring_edges(n) -> tuple[int, list[tuple[int, int]]]:
    assert n >= 3       # ring graph doesn't make sense in these cases
    _, ring = line_edges(n)
    ring.append((0, n - 1))
    return n, ring


def edges_to_topology(
        graph: tuple[int, list[tuple[int, int]]]) -> dict[int, list[int]]:
    n, edges = graph
    topology = {i: [] for i in range(n)}
    for k, v in edges:
        topology[k].append(v)
        topology[v].append(k)
    return topology


# Find the max distance between any two points
def getDiameter(topology: dict[int, list[int]]) -> int:
    a_matrix = np.zeros((len(topology), len(topology)))
    for node, neighbors in topology.items():
        a_matrix[node, neighbors] = 1

    diameter = scipy.sparse.csgraph.shortest_path(a_matrix, unweighted=True)
    if np.isinf(diameter).any():
        raise ValueError("The graph is disconnected")
    return int(diameter.max())
