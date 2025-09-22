# simulation_daa/utils/graph_utils.py
def build_intra_edges_copies(start_ts, end_ts, P, N):
    base = {
        i * N + j: (i * N + ((j + 1) % N), i * N + ((j - 1 + N) % N))
        for i in range(P) for j in range(N)
    }
    out = {}
    for t in range(start_ts, end_ts):
        out[t] = {u: set(vs) for u, vs in base.items()}
    return out

def make_edges_bidirectional(edge_dict):
    out = {}
    for u, vs in (edge_dict or {}).items():
        for v in vs:
            if u == v: continue
            out.setdefault(u, set()).add(v)
            out.setdefault(v, set()).add(u)
    return out
