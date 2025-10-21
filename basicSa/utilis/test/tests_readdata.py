import basicSa.utilis.readdata as readdata





if __name__ == '__main__':
    stationpath = "/home/yfh/Desktop/Data/stations"
    stationangle = 20
    stationsnodes = readdata.readstation(stationpath, stationangle)  # 获取地面站数据

    # 读取卫星初始位置
    satpath = "/home/yfh/Desktop/Data/sats"
    satangle = 20




    satsnodes = readdata.readsats_multi(satpath, satangle)  # 获取卫星数据
    #print(stationsnodes[0])

    # wo alse need add the trackangle and the RAAN for the basicSa
    trackangle = 89

    N=20
    P=2
    #sometimes ,the RAAN always equre to

    for i in range(P):
        for j in range(N):
            number = i*N+j
            RAAN=10.2*(i+1)
            readdata.add_number_RAAN( satsnodes[number],RAAN,trackangle,number)
            # satsnodes[number].RAAN=10.2*(i+1)
            # satsnodes[number].number = trackangle
    print(satsnodes[2])
