# command_line.py
import argparse

import  math
import basicSa.allrun.readstation as readstation

import basicSa.allrun.readsats as readsats
import basicSa.allrun.calculate_angular as calculate_angular

def main():
    dir_path = '/home/yfh/Desktop/Data/stations2'

    stations_nodes = readstation.load_station(dir_path)

    dir_path = '/home/yfh/Desktop/Data/onehun_ecef'
    satsangle = 45
    track_angle = math.radians(89)
    P = 10
    N = 10

    sats_nodes = readsats.load_trajectory_filenew(dir_path, satsangle, track_angle, P, N)

    s1 =sats_nodes[0]
    max_time  = len(s1)
    print("1")

    for i in range(max_time-1):

        sat_nodes_time = []
        vecs_base = []
        vecs_next = []


        for k in range(P*N):
            sat_nodes_time.append(sats_nodes[k][i] )
            vecs_base.append([sats_nodes[k][i].x,sats_nodes[k][i].y,sats_nodes[k][i].z])


            vecs_next.append([sats_nodes[k][i+1].x,sats_nodes[k][i+1].y,sats_nodes[k][i+1].z])

        vector_current_deleta = []
        vecs_deleta_next = []
        for u in range((P-1)*N):
            start =u
            start_planid = u//N
            start_orbitid = u%N
            mubiaoid =(start_planid+1)*start_orbitid

            vector_current_deleta.append(vecs_base[mubiaoid]-vecs_base[u])
            vecs_deleta_next.append(vecs_next[mubiaoid]-vecs_next[u])

        # every seconds


        for j in range((P-1)):


            oribit_id = j
            start = oribit_id*N
            end =  start+N-1
            RAAN = sat_nodes_time[start].RAAN
            angular = calculate_angular.calculate_angular_velocity( RAAN,
                                                                    track_angle,
                                                                    i,vecs_base,
                                                                    vecs_next,vector_current_deleta,vecs_deleta_next)



        print("2")





if __name__ == "__main__":
    main()
