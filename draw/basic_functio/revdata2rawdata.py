
import  draw.read_snap_xml as read_snap_xml


def revedge2rawedge(edge_by_step,off_sets):
    modify_edge_by_step = {}
    for step, src_to_dsts in edge_by_step.items():
        for src, dsts in src_to_dsts.items():  # dsts是个集合
            # 只取集合中的第一个（如果你确认每个集合只有一个dst的话）
            if step==1:
                if src==44:
                    print(src)
            dst = next(iter(dsts))

            modify_src = read_snap_xml.rev_modify_data(step, src, off_sets)
            modify_dst = read_snap_xml.rev_modify_data(step, dst, off_sets)

            # 先初始化
            if step not in modify_edge_by_step:
                modify_edge_by_step[step] = {}
            modify_edge_by_step[step].setdefault(modify_src, set()).add(modify_dst)


    return modify_edge_by_step


