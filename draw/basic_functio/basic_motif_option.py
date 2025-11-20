# # 4-5
# import  math
# if option == 0:
#     nextnode_x = x + 1
#     nextnode_y = y
#
# # 4-8
# elif option == 1:
#
#     nextnode_x = x + 1
#     nextnode_y = (y - 1 + N) % N
#
# #
# # 4-6
# elif option == 2:
#
#     nextnode_x = x + 2
#     nextnode_y = y
# # 4-2
# elif option == 4:
#
#     nextnode_x = x + 1
#     nextnode_y = (y + 1) % N
# # 4-9
# elif option == 5:
#     nextnode_x = x + 2
#     nextnode_y = (y - 1 + N) % N

# here x1<x2

def getoption(x1,y1,x2,y2,N):
    if x2-x1==1 and y2-y1==0:
        return 0
    elif x2-x1==1 and  y2==(y1 - 1 + N) % N :
        return 1
    elif x2-x1==2 and y2==y1:
        return 2
    elif x2-x1==1 and y2==(y1 + 1) % N:
        return 4
    elif x2-x1==2 and y2==(y1 - 1 + N) % N:
        return 5

