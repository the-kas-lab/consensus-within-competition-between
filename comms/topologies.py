"""Controlled graph regimes.

The dynamic disk graph is the primary model; these fixed topologies exist so
protocol properties can be tested against a graph whose diameter is known in
advance. "One hop per round" is only checkable on a line, and "components may
disagree and the disagreement is logged rather than silently repaired" is only
checkable on a partition.

Each builder returns an adjacency set of frozensets -- unordered pairs, because
every link here is symmetric.
"""


def complete(n):
    return {frozenset((i, j)) for i in range(n) for j in range(i + 1, n)}


def line(n):
    """0 - 1 - 2 - ... - (n-1). Diameter n-1, the slowest useful graph and the
    one that makes one-hop-per-round observable."""
    return {frozenset((i, i + 1)) for i in range(n - 1)}


def ring(n):
    if n < 3:
        return line(n)
    return {frozenset((i, (i + 1) % n)) for i in range(n)}


def partition(n, split=None):
    """Two complete components with no edge between them.

    ``split`` defaults to n//2, giving the plan's "2+2 partition" at n=4. The
    point is a graph where a belief provably cannot cross, so a test can assert
    that the two halves *stay* disagreeing rather than quietly converging
    through some path that should not exist.
    """
    split = n // 2 if split is None else int(split)
    left = range(split)
    right = range(split, n)
    edges = {frozenset((i, j)) for i in left for j in left if i < j}
    edges |= {frozenset((i, j)) for i in right for j in right if i < j}
    return edges


def disk(positions, r_comm):
    """Distance-thresholded graph from live positions -- the dynamic case.

    ``positions`` maps agent id -> length-2 array-like.
    """
    import numpy as np

    ids = sorted(positions)
    edges = set()
    for a_index, i in enumerate(ids):
        for j in ids[a_index + 1:]:
            d = float(np.linalg.norm(np.asarray(positions[i], float) - np.asarray(positions[j], float)))
            if d <= r_comm:
                edges.add(frozenset((i, j)))
    return edges


def neighbours(edges, agent):
    out = set()
    for edge in edges:
        if agent in edge:
            pair = tuple(edge)
            out.add(pair[1] if pair[0] == agent else pair[0])
    return out


def components(edges, agents):
    """Connected components, as a list of sorted agent lists.

    Used by the metrics asks for (largest-component fraction,
    partition count) and by tests that assert a belief did not cross a cut.
    """
    remaining = set(agents)
    found = []
    while remaining:
        root = min(remaining)
        stack, seen = [root], {root}
        while stack:
            node = stack.pop()
            for nbr in neighbours(edges, node):
                if nbr in remaining and nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        found.append(sorted(seen))
        remaining -= seen
    return found


def diameter(edges, agents):
    """Longest shortest-path within the largest component, or None if the graph
    is disconnected in a way that makes a single diameter meaningless.

    Reported per wants "graph diameter while connected" --
    so a disconnected graph returning None is the honest answer, not 0.
    """
    agents = sorted(agents)
    comps = components(edges, agents)
    if len(comps) != 1:
        return None
    best = 0
    for source in agents:
        dist = {source: 0}
        queue = [source]
        while queue:
            node = queue.pop(0)
            for nbr in neighbours(edges, node):
                if nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        best = max(best, max(dist.values()))
    return best
