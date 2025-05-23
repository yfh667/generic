import random
import math
import matplotlib.pyplot as plt
import random
import numpy as np
from networkx.classes import neighbors
import genaric.chrom2adjact as c2a
import ga.graphalgorithm.fcnfp_multi as fcnfp_multi

import genaric2.initialize_individual as initialize_individual
import draw.snapshotf_romxml as snapshotf_romxml
import copy

import genaric2.initiallink as initiallink
import genaric.chrom2adjact as c2a

import genaric.plotgraph as plotgraph

import ga.graphalgorithm.adjact2weight as a2w
import os
import draw.snapshotf_romxml as snapshotf_romxml
import graph.drawall as drawall
import genaric2.tegnode as tegnode

import graph.time_2d2 as time_2d
# --- Example Usage ---
import genaric2.distinct_initial as distinct_initial
import genaric2.action_table as action_table
# 定义目标函数
import genaric2.cross as cross
# 参数设置
#topology
N = 7
P=9
distinct = [[17,18,24,25],[36,37,43]]

SOURCES = {17: 70, 18: 70, 24: 70, 25: 70}
SINKS = distinct[1]

inter_link_bandwidth = 50
intra_link_bandwidth = 100

cost =1
population_size = 50  # 种群大小
generations = 5  # 最大迭代代数
mutation_rate = 0.2  # 变异概率
crossover_rate = 0.25  # 交叉概率
chromosome_length = N*(P-1)  # 二进制染色体长度（22位）



# 将二进制染色体转换为十进制数值
def decode_chromosome(chromosome):
    base_adjacency_list = c2a.base_chrom2adjacent(chromosome, N, P)
    adjacency_list = c2a.full_adjacency_list(base_adjacency_list, N, P)
    return adjacency_list



def get_fixed_right_neighbor(N,distinct):
    fixed_right_neighbor = []
    for dist in distinct:  # 假设 distinct 是一个包含多个节点集合的列表
        for node in dist:
            i_node = node // N  # 计算行号
            j_node = node % N  # 计算列号

            # 检查右邻居是否存在（i_node < N-1 表示不在最右列）
            if i_node < N - 1:
                right_neighbor = (i_node + 1) * N + j_node

                # 如果右邻居也在当前集合 dist 中，则建立连接
                if right_neighbor in dist:
                    # 初始化 node 的邻接集合（如果不存在）
                    fixed_right_neighbor.append(node)
    return fixed_right_neighbor




# 初始化种群
def initialize_population(P,N,T,nodes,setuptime):
    chromosome = []
    for i in range(population_size):
        nodes_copy = copy.deepcopy(nodes)  # 深复制nodes

        chromosome.append(   initialize_individual.initialize_individual(P, N, T, nodes_copy,setuptime))

    return chromosome

   # return [random.sample(range(chromosome_length), chromosome_length) for _ in range(population_size)]


# 计算种群适应度
def fitness_function(population):
    indictors = []
    decoded_values = [decode_chromosome(ind) for ind in population]
    for adjacency_list in decoded_values:

        full_adjacency_list = c2a.full_adjacency_list(adjacency_list, N, P)


        edge = a2w.adjacent2edge(full_adjacency_list, N, inter_link_bandwidth, intra_link_bandwidth, cost)




        # 使用新函数求解
        multi_result = fcnfp_multi.solve_multi_source_sink_with_super_nodes(
            edges_data=edge,
            sources=SOURCES,
            sinks=SINKS
        )
        indictor = multi_result['total_fixed_cost']







        indictors.append(indictor)
    #  print(indictor)

    # Rastrigin函数（多峰函数，常用于测试优化算法）

    return indictors




def filter_and_replenish(population, fitness):
    """
    过滤掉适应度为inf的个体，并用最优个体补充

    参数:
        population: 当前种群列表
        fitness: 对应的适应度列表

    返回:
        过滤并补充后的新种群
    """
    # 过滤掉inf的个体
    valid_pop = []
    valid_fitness = []
    for ind, fit in zip(population, fitness):
        if not math.isinf(fit):  # 检查是否是inf
            valid_pop.append(ind)
            valid_fitness.append(fit)

    # 计算需要补充的数量
    num_removed = len(population) - len(valid_pop)

    if num_removed > 0:
        # 找到适应度最好的个体(最小值)
        best_fitness = min(valid_fitness)
        best_individuals = [ind for ind, fit in zip(valid_pop, valid_fitness)
                            if fit == best_fitness]

        # 随机选择最优个体进行补充
        for _ in range(num_removed):
            valid_pop.append(random.choice(best_individuals))

    return valid_pop


# 选择操作（轮盘赌选择）
def selection(population):
    fitness = fitness_function(population)
 #   valid_pop = filter_and_replenish(population, fitness)
  #  fitness = fitness_function(valid_pop)
   # here we need delete some dead individual

    total_fitness = sum(fitness)


    probabilities = [f / total_fitness for f in fitness]
    cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

    new_population = []
    for _ in range(population_size):
        r = random.random()
        for i, cp in enumerate(cumulative_probabilities):
            if r <= cp:
                new_population.append(population[i])
                break
    return new_population


# 变异操作
def mutate(individual,fixed_right_neighbor):
    individual_copy = individual.copy()  # 创建副本
    #random_index = random.sample(range(len(individual_copy)), 1)
    random_index = random.choice(range(len(individual_copy)))

    if random_index in fixed_right_neighbor:
        return individual_copy

    colid =random_index//N
    rowid = random_index%N
   # candidates = [{(i - 1) % N, i % N, (i + 1) % N} for i in range(N)]
    random_neighbors_id = random.choice(range(3))  # 从 0, 1, 2 中随机选

    #random_neighbors_id =  random.choice(3)
    #neighbors_row_id = candidates[rowid][random_neighbors_id]

    neighbors_id = (colid+1)*N + (random_neighbors_id-1+rowid)%N

    individual_copy[random_index]=neighbors_id


# 处理冲突的
    for  j in range(N):
        nodeid = colid*N + j
        if nodeid !=random_index:
            if   individual_copy[nodeid] ==individual_copy[random_index]:
                 individual_copy[nodeid] =-1

  #  tmp = individual_copy[random_numbers[0]]
  #  individual_copy[random_numbers[0]] = individual_copy[random_numbers[1]]
   # individual_copy[random_numbers[1]] = tmp


    return individual_copy


# 遗传算法主流程
def genetic_algorithm():
    ##1.我们先获得原始的卫星分布数据，主要是热点区域的拓扑序列数据
    N = 10
    P = 10
    start_ts = 1500
    end_ts = 1523

    dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"

    regions_to_color = {}
    # Iterate over the time steps
    region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
    target_time_step = len(region_satellite_groups)
    T = target_time_step
    for i in range(len(region_satellite_groups)):
        region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
        u = region_satellite_group[0]
        v = region_satellite_group[3]
        o = [u, v]
        regions_to_color[i] = o  # Corrected append to dictionary assignment

    ##2. 完成对热点区域预建链安排

    T = target_time_step
    setuptime = 2

    ## nodes是所个体的基础骨架
    # left_port 是所有个体随机生成的限制，可以认为，每个个体必须要满足同样的核心结构，就类似，每个个体可以长相不一样，基本上都要相同数量的骨头
    nodes = distinct_initial.distinct_initial(P, N, T, setuptime, regions_to_color)



    population =initialize_population(P,N,T,nodes,setuptime)
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []
    fixed_right_neighbor = get_fixed_right_neighbor(N,distinct)


    for generation in range(generations):
        print(f"generation {generation}")
        # 选择操作

        # 交叉和变异
        next_population = []
        # we need select the corss individial
        random_array = [random.uniform(0, 1) for _ in range(len(population))]
        crossarrary = []
        for i in range(len(random_array)):
            if (random_array[i] < crossover_rate):
                crossarrary.append(i)
                # here we get the zajiao duixiang shuzu

        for i in range(0, len(crossarrary), 2):
            if (i != len(crossarrary) - 1):
                parent1_id = crossarrary[i]

                parent2_id = crossarrary[i + 1]
                parent1 = population[parent1_id]
                parent2 = population[parent2_id % len(population)]
                child1, child2 = cross.crossover(parent1, parent2,P,N,T,setuptime)

        ## 变异, 我们针对的是父代变异
        mutate_random_array = [random.uniform(0, 1) for _ in range(len(population))]
        mutate_arrary = []
        for i in range(len(mutate_random_array)):
            if (mutate_random_array[i] < mutation_rate):
                mutate_arrary.append(i)

        # here we need mutate the raw population
        mutate_population = []
        for i in range(len(mutate_arrary)):
            individual = population[mutate_arrary[i]]
            mutate_individual = mutate(individual,distinct)

            mutate_population.append(mutate_individual)

        population = population + next_population + mutate_population

        population = selection(population)

        fitness = fitness_function(population)

        best_idx = fitness.index(min(fitness))
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_solution = decode_chromosome(population[best_idx])

        fitness_history.append(best_fitness)

        print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}, Best Solution = {best_solution},")

    return best_solution, best_fitness, fitness_history


# 执行遗传算法
best_x, best_y, fitness_history = genetic_algorithm()
print(f"Optimal solution: x = {best_x}, f(x) = { best_y}")

# 绘制适应度历史曲线
reciprocal_history = [1.0 / fitness for fitness in fitness_history]
plt.figure()
plt.plot(range(len(reciprocal_history)), reciprocal_history, marker='o', label="Best Fitness")
plt.title("Fitness History Over Generations")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.legend()
plt.grid()
plt.show()

plotgraph.plot_graph_with_auto_curve_distinct(best_x,N,P,distinct)

