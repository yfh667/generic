
import genaric2.fitness.fitness as fitnessfuc

import random

import genaric2.mutation2.mutate1 as mutate1
import genaric2.mutation2.mutate2 as mutate2
import genaric2.mutation2.mutate3 as mutate3
import copy




def mutate_f(individual,regions_to_color,inter_link_bandwidth, intra_link_bandwidth, cost,P, N, T,setuptime ):




   # individual_copy = individual.copy()  # 创建副本
    individual_copy = copy.deepcopy(individual)

    fitness, indictors_seq = fitnessfuc.calculate_fitness_onedividual(individual_copy,regions_to_color, inter_link_bandwidth, intra_link_bandwidth, cost,P, N, T )


    min_value = min(indictors_seq)
    min_index = indictors_seq.index(min_value)


    # here we need muate the individual[_,_,min_index] min_index chromso


# first we need detele the region-to-color
    region_distinct= regions_to_color[min_index]





    delete_nodes = []
    for distinct in region_distinct:
        for nodes in distinct:
            x = nodes // N
            y = nodes % N

            coordinate = (x, y, min_index)

            right_neighbor = individual_copy[coordinate].rightneighbor

            if right_neighbor:
                if individual_copy[right_neighbor].asc_nodes_flag==True:
                    delete_nodes.append(nodes)

            # if individual_copy[coordinate].asc_nodes_flag==True:
            #     delete_nodes.append(nodes)


    # 获取所有合法的 index，去除 delete_nodes
    all_indices = set(range(P * N))
    delete_indices = set(delete_nodes)
    valid_indices = list(all_indices - delete_indices)

    # 判断是否还有可用 index
    if not valid_indices:
        raise ValueError("No valid indices left for mutation after excluding delete_nodes.")

    # 从剩余合法索引中随机选择一个
    mutate_index = random.choice(valid_indices)



    x_mutate_index = mutate_index // N
    y_mutate_index = mutate_index % N


#
    nowdistinct = []
    for distinct in region_distinct:
        for nodes in distinct:
            x = nodes // N
            y = nodes % N
            if nodes==(x_mutate_index,y_mutate_index):
                nowdistinct=distinct
# mutate_index = random.randint(0, P * N - 1)
#
#     x_mutate_index = mutate_index // N
#     y_mutate_index=mutate_index%N

# so we nned mutate individual[x_mutate_index,y_mutate_index,min_index]
    mutate_node = (x_mutate_index,y_mutate_index,min_index)
    if individual_copy[mutate_node].state==0:
        mutate1.establishment_mutate(mutate_node, individual_copy, nowdistinct,P, N, T, setuptime)
    elif individual_copy[mutate_node].state==2:
        mutate2.maintenance_mutate(mutate_node, individual_copy, P, N, T, setuptime)
    elif  individual_copy[mutate_node].state==-1:
        mutate3.disconenct_mutate(mutate_node, individual_copy, P, N, T, setuptime)
    return individual_copy,mutate_node



def mutate_f_test(individual,regions_to_color,inter_link_bandwidth, intra_link_bandwidth, cost,P, N, T,setuptime ):




   # individual_copy = individual.copy()  # 创建副本
    individual_copy = copy.deepcopy(individual)

#     fitness, indictors_seq = fitnessfuc.calculate_fitness(individual_copy,regions_to_color, inter_link_bandwidth, intra_link_bandwidth, cost,P, N, T )
#
#
#     min_value = min(indictors_seq)
#     min_index = indictors_seq.index(min_value)
#
#
#     # here we need muate the individual[_,_,min_index] min_index chromso
#
#     mutate_index = random.randint(0, P * N - 1)
#
#     x_mutate_index = mutate_index // N
#     y_mutate_index=mutate_index%N
#
# # so we nned mutate individual[x_mutate_index,y_mutate_index,min_index]
#     mutate_node = (x_mutate_index,y_mutate_index,min_index)


    mutate_node=(6, 8, 6)


    region_distinct = regions_to_color[mutate_node[2]]

    nowdistinct = []
    for distinct in region_distinct:
        for nodes in distinct:
            x = nodes // N
            y = nodes % N
            if nodes==(mutate_node[0],mutate_node[1]):
                nowdistinct=distinct

    if individual_copy[mutate_node].state==0:
        mutate1.establishment_mutate(mutate_node, individual_copy,nowdistinct, P, N, T, setuptime,1)
    elif individual_copy[mutate_node].state==2:
        mutate2.maintenance_mutate(mutate_node, individual_copy, P, N, T, setuptime)
    elif  individual_copy[mutate_node].state==-1:
        mutate3.disconenct_mutate(mutate_node, individual_copy, P, N, T, setuptime)
    return individual_copy,mutate_node

