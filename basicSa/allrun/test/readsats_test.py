import basicSa.allrun.readsats as readsats
dir_path = '/home/yfh/Desktop/Data/onehun_ecef'
satsangle = 45
track_angle = 89
P=10
N=10
satsnodes = readsats.load_trajectory_filenew(dir_path,satsangle,track_angle,P,N)
s1 =satsnodes[0]
