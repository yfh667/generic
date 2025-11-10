# 首先定义一个类，要有__init__
class tegnode:
    def __init__(self,asc_nodes_flag,rightneighbor,leftneighbor,state,importance):
        self.asc_nodes_flag = asc_nodes_flag
        self.rightneighbor = rightneighbor
        self.leftneighbor = leftneighbor
        #state:-2 : can't be used to setup link, but can be used to work
        #state -1: free
        #state 0 :setting link
        #state 1: working

        self.state = state
        self.importance =importance

    def __repr__(self):
        return f"tegnode(asc_nodes_flag={self.asc_nodes_flag}, rightneighbor={self.rightneighbor}, leftneighbor={self.leftneighbor}, state={self.state}),importance={self.importance})"


class tegnode_new:
    def __init__(self,asc_nodes_region_id,rightneighbor,leftneighbor,state,importance):
        self.asc_nodes_region_id = asc_nodes_region_id
        # if asc_nodes_region_id==-1,it means ,it is not in any region
        # the region ,always from 0-M,which always in our assumption
        self.rightneighbor = rightneighbor
        self.leftneighbor = leftneighbor
        #state:-2 : can't be used to setup link, but can be used to work
        #state -1: free
        #state 0 :setting link
        #state 1: working

        self.state = state
        self.importance =importance

    def __repr__(self):
        return f"tegnode(asc_nodes_region_id={self.asc_nodes_region_id}, rightneighbor={self.rightneighbor}, leftneighbor={self.leftneighbor}, state={self.state}),importance={self.importance})"

class tegnode_complete:
    def __init__(self,asc_nodes_region_id,rightneighbor,leftneighbor,left_state,right_state):
        self.asc_nodes_region_id = asc_nodes_region_id
        # if asc_nodes_region_id==-1,it means ,it is not in any region
        # the region ,always from 0-M,which always in our assumption
        self.rightneighbor = rightneighbor
        self.leftneighbor = leftneighbor



        #state 0 :setting link
        #state 1: working
        self.left_state = left_state
        self.right_state = right_state


    def __repr__(self):
        return f"tegnode(asc_nodes_region_id={self.asc_nodes_region_id}, rightneighbor={self.rightneighbor}, leftneighbor={self.leftneighbor}, left_state={self.left_state},right_state={self.right_state}"



# 建议：约定 -1 表示“未设置/未知”
STATE_SETTING = 0   # 建链中
STATE_WORKING = 1   # 工作中
STATE_UNKNOWN = -1  # 未知/未设置

class tegnode_new:
    """
    asc_nodes_region_id: 所在区域ID；-1 表示不在任何区域
    rightneighbor / leftneighbor: 右/左邻居节点ID；-1 表示无
    left_state / right_state: 0=建链中, 1=工作中, -1=未知
    status: 节点总体状态；业务自定义，-1=未知
    type: 节点类型；业务自定义，-1=未知
    timelast: 最近一次状态持续的时长或时间戳；-1=未知
    """

    def __init__(
        self,
        asc_nodes_region_id: int = -1,
        rightneighbor: int = -1,
        leftneighbor: int = -1,
        left_state: int = STATE_UNKNOWN,
        right_state: int = STATE_UNKNOWN,
        node_type: int = -1,
        timelast: int = -1
    ):
        self.asc_nodes_region_id = asc_nodes_region_id
        # 若 asc_nodes_region_id == -1，表示不属于任何区域（区域通常从 0..M）
        self.rightneighbor = rightneighbor
        self.leftneighbor = leftneighbor

        # state: 0=建链中，1=工作中，-1=未知
        self.left_state = left_state
        self.right_state = right_state


        self.node_type = node_type   # 如需避免覆盖内置名，可改为 node_type
        self.timelast = timelast

    def __repr__(self) -> str:
        return (
            "tegnode("
            f"asc_nodes_region_id={self.asc_nodes_region_id}, "
            f"rightneighbor={self.rightneighbor}, "
            f"leftneighbor={self.leftneighbor}, "
            f"left_state={self.left_state}, "
            f"right_state={self.right_state}, "
            f"type={self.node_type}, "
            f"timelast={self.timelast}"
            ")"
        )