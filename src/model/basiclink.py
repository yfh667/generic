def make_edges_bidirectional(edge_dict):
    """{basicSa: set(dsts)} -> 双向"""
    new_edges = {}
    for src, dsts in edge_dict.items():
        for dst in dsts:
            new_edges.setdefault(src, set()).add(dst)
            new_edges.setdefault(dst, set()).add(src)
    return new_edges
