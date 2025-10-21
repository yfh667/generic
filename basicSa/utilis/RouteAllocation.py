import basicSa.utilis.Node as  Nodepy


# here we assume that we know the path and we have set up the link allocation
#for example path =  [0,12,32,2]
#
#our goal
# [12, 0]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [2, 0]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]

#our now
# [12, 0]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]
# [-1, -1]

def RouteAllocation(path,srcid,dstid,srcsat:Nodepy.Node):
    srcport =  GetLinkPort(path,srcid,dstid,srcsat)
    print(srcport)
    SetRouteAllocation(srcport, dstid, srcsat)

def SetRouteAllocation(sat1port, sat2id, satellite1:Nodepy.Node):

    # 假设 satellite1 是 Node 类的一个实例

    for i in range(5, 15):
        if(satellite1.get_value(i)==[sat2id, sat1port]):
            break
        if(satellite1.get_value(i)==[-1,-1]):
            satellite1.set_value(i, [sat2id, sat1port])  # 替换 your_value_1 和 your_value_2 为实际的值
            break

#path =  [0,12,32,2]
#basicSa = 0,dst =2
#srcid = 0
#dstid = 2
def GetLinkPort(path,srcid,dstid,srcsat:Nodepy.Node):
    # we find the 12 index in the path :here it is 1
    try:
        src_index = path.index(srcid)
        print(f"The index of srcid {srcid} in path is {src_index}")
    except ValueError:
        print(f"srcid {srcid} not found in path")
    #here nexthopid is 16
    nexthopid = path[src_index+1]
    for i in range(0, 6):
        portid = i
        linkid =srcsat.get_value(i)  #[12,0],nextsat id is 3,and link to his port2,and nwo sat use port i
        if(linkid[0]==nexthopid):
            srcport = i
            break
    return srcport






