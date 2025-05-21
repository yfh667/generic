# 首先定义一个类，要有__init__
class tegnode:
    def __init__(self,asc_nodes_flag,rightneighbor,state):
        self.asc_nodes_flag = asc_nodes_flag
        self.rightneighbor = rightneighbor
        #state -1: free
        #state 0 :setting link
        #state 1: working
        self.state = state

