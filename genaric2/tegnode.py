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

