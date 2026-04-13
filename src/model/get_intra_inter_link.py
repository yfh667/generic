
def parse_path_links(path_str, N=36):
    """
    输入: '86->85->120->119->...'
    输出: (intra_links, inter_links)
        intra_links: [(86,85), ...]   同轨链路（orbit 相同）
        inter_links: [(85,120), ...]  异轨链路（orbit 不同）
    """
    if not path_str or path_str == "":
        return [], []

    nodes = [int(x) for x in path_str.split("->")]
    intra_links = []
    inter_links = []

    for i in range(len(nodes) - 1):
        src, dst = nodes[i], nodes[i + 1]
        src_orbit = src // N
        dst_orbit = dst // N

        if src_orbit == dst_orbit:
            intra_links.append((src, dst))
        else:
            inter_links.append((src, dst))

    return intra_links, inter_links
