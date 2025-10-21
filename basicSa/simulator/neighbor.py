def _get_right_neighbor2( N,P,sat_id, i):
    # i=1 to 6
    """获取右邻居ID"""
    if i == 1:
        orbit = sat_id // N + 1
        plane = (sat_id + 1) % N
        if orbit < P:
            return orbit, plane, (orbit) * N + plane
    elif i == 2:
        orbit = sat_id // N + 1
        plane = (sat_id) % N
        if orbit < P:
            return orbit, plane, (orbit) * N + plane
    elif i == 3:
        orbit = sat_id // N + 1
        plane = (sat_id + N - 1) % N
        if orbit < P:
            return orbit, plane, (orbit) * N + plane
    elif i == 4:
        orbit = sat_id // N + 1
        plane = (sat_id + N - 2) % N
        if orbit < P:
            return orbit, plane, (orbit) * N + plane
    elif i == 5:
        orbit = sat_id // N + 2
        plane = (sat_id) % N
        if orbit < P:
            return orbit, plane, (orbit) * N + plane

    # elif i==6:
    #
    #     orbit = sat_id // self.N+2
    #     plane = (sat_id-1+ self.N) % self.N
    #     if orbit < self.P :
    #         return orbit,plane,(orbit ) * self.N + plane

    else:
        orbit = sat_id // N + 2
        plane = (sat_id - 2 + N) % N
        if orbit < P:
            return orbit, plane, (orbit) * N + plane
    #   print(1)
    return None


def _get_left_neighbor2(satid, N,tag):
    satid_planid = satid //N
    orbitid = satid % N
    if tag==0:
        L_plane = satid_planid-1
        L_orbitid = (orbitid+2)%N
        return L_plane, L_orbitid,L_plane*N+L_orbitid
    elif tag==1:
        L_plane = satid_planid-1
        L_orbitid = (orbitid+1)%N
        return L_plane, L_orbitid,L_plane*N+L_orbitid
    elif tag==2:
        L_plane = satid_planid-1
        L_orbitid = (orbitid)%N
        return L_plane, L_orbitid,L_plane*N+L_orbitid

    elif tag==3:
        L_plane = satid_planid-1
        L_orbitid = (orbitid-1+N)%N
        return L_plane, L_orbitid,L_plane*N+L_orbitid
    elif tag==4:
        L_plane = satid_planid-2
        L_orbitid = (orbitid+2)%N
        return L_plane, L_orbitid,L_plane*N+L_orbitid
    elif tag==5:
        L_plane = satid_planid-2
        L_orbitid = (orbitid)%N
        return L_plane, L_orbitid,L_plane*N+L_orbitid


def findneighbor_id(now,neighbor,N):
    nowplane = now//N
    noworbitid = now%N

    neighbor_plane = neighbor // N
    neighbor_orbitid = neighbor % N
    if nowplane==neighbor_plane-1 :
        if neighbor_orbitid==noworbitid+1:
            return 1
        elif neighbor_orbitid==noworbitid:
            return 2
        elif neighbor_orbitid==noworbitid-1:
            return 3
        elif neighbor_orbitid==noworbitid-2:
            return 4
    elif nowplane==neighbor_plane-2:
        if neighbor_orbitid==noworbitid:
            return 5
        elif neighbor_orbitid==noworbitid-2:
            return 6
    elif nowplane==neighbor_plane+1:
        if neighbor_orbitid == noworbitid + 2:
            return 7
        elif neighbor_orbitid == noworbitid+1:
            return 8
        elif neighbor_orbitid == noworbitid :
            return 9
        elif neighbor_orbitid == noworbitid - 1:
            return 10
    else:
        if neighbor_orbitid==noworbitid+2:
            return 11
        elif neighbor_orbitid==noworbitid:
            return 12
