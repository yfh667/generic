
import  draw.read_snap_xml as read_snap_xml


# 输入的是修改后的边，输出的是原始边
# def revedge2rawedge(edge_by_step,off_sets):
#     modify_edge_by_step = {}
#     for step, src_to_dsts in edge_by_step.items():
#         for src, dsts in src_to_dsts.items():  # dsts是个集合
#             # 只取集合中的第一个（如果你确认每个集合只有一个dst的话）
#
#
#
#             modify_src = read_snap_xml.rev_modify_data(step, src, off_sets)
#             modify_dst = read_snap_xml.rev_modify_data(step, dst, off_sets)
#
#             # 先初始化
#             if step not in modify_edge_by_step:
#                 modify_edge_by_step[step] = {}
#             modify_edge_by_step[step].setdefault(modify_src, set()).add(modify_dst)
#
#
#     return modify_edge_by_step


# 输入的是修改后的边，输出的是原始边
def revedge2rawedge(modify_edge_by_step, off_sets):

    #
    raw_edges_by_step = {}

    for step, edges in modify_edge_by_step.items():  # 一定要遍历raw_edges_by_step！
        raw_edges_by_step[step] = {}

        for src, dsts in edges.items():
            modify_src = read_snap_xml.rev_modify_data(step, src, off_sets)
            for dst in dsts:
                modify_dst = read_snap_xml.rev_modify_data(step, dst, off_sets)
                raw_edges_by_step[step].setdefault(modify_src, set()).add(modify_dst)

    return raw_edges_by_step
# 输入的是原始边，输出的是修改后的边
def edge2revedge(raw_edges_by_step,off_sets):

    #
    modify_edges_by_step = {}

    for step, edges in raw_edges_by_step.items():  # 一定要遍历raw_edges_by_step！
        modify_edges_by_step[step] = {}

        for src, dsts in edges.items():
            modify_src = read_snap_xml.modify_data(step, src, off_sets)
            for dst in dsts:
                modify_dst = read_snap_xml.modify_data(step, dst, off_sets)
                modify_edges_by_step[step].setdefault(modify_src, set()).add(modify_dst)

    return modify_edges_by_step