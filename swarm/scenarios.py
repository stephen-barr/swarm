def line_edges(n) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def complete_edges(n) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(n) if i < j]


def ring_edges(n) -> list[tuple[int, int]]:
    assert n >= 3       # ring graph doesn't make sense in these cases
    ring = line_edges(n)
    ring.append((0, n - 1))
    return ring


def edges_to_topology(n: int, edges: list) -> dict:
    topology = {i: [] for i in range(n)}
    for k, v in edges:
        topology[k].append(v)
        topology[v].append(k)
    return topology
