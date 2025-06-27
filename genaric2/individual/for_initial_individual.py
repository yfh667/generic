import random
import math
import matplotlib.pyplot as plt
import random

import genaric2.writetoxml as writetoxml
import genaric2.initialize_individual as initialize_individual

import  genaric2.mutation3_probability.mutate as mutate
import  genaric2.mutation.mutate as mutate_raw

import os
import draw.snapshotf_romxml as snapshotf_romxml

# --- Example Usage ---
import genaric2.distinct_initial as distinct_initial

# 定义目标函数
import genaric2.cross.cross2 as cross

import genaric2.fitness.fitness as fitnessfuc
import genaric2.TopoSeqValidator as TopoSeqValidator

# 参数设置
#topology_prof
#this function ,based on generic2.





inter_link_bandwidth = 50
intra_link_bandwidth = 100

cost =1
population_size = 200  # 种群大小
generations = 300  # 最大迭代代数
mutation_rate = 0.3  # 变异概率
crossover_rate = 0.15  # 交叉概率

N = 10
P = 10
start_ts = 1500
end_ts = 1523
setuptime = 2

dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"

regions_to_color = {}
# Iterate over the time steps
region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
target_time_step = len(region_satellite_groups)
T = target_time_step
for i in range(T):
    region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
    u = region_satellite_group[0]
    v = region_satellite_group[3]
    o = [u, v]
    regions_to_color[i] = o  # Corrected append to dictionary assignment








# 初始化种群
def initialize_population(P,N,T,region,setuptime):
    chromosome = []
    for i in range(population_size):
    #    nodes_copy = copy.deepcopy(nodes)  # 深复制nodes


        chromosome.append(   initialize_individual.initialize_individual_region(region,P, N, T, setuptime))




        flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(chromosome[i], P, N, T,
                                                                                               regions_to_color,
                                                                                               setuptime)

        if flag1 == 0:
            print(f" para cuowu {i}")

            writetoxml.nodes_to_xml(chromosome[i], "E:\\code\\data\\2\\produce_individual.xml")

            os._exit(1)
            try:
                sys.exit("程序终止：存在非法拓扑结构。")
            except SystemExit as e:
                raise e  # 强制抛出，不让 IDE 忽略

    return chromosome

   # return [random.sample(range(chromosome_length), chromosome_length) for _ in range(population_size)]




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



import sys
def selection(population, population_size,setuptime):
    # 原始适应度：值越小越好
    fitness, _ = fitnessfuc.fitness_function(population, regions_to_color, intra_link_bandwidth, inter_link_bandwidth, cost, P, N, T)

    # 防止 fitness 值全为 0 或过小导致除法出错，做一次反向变换（越小越好 → 越大越好）
    max_fit = max(fitness)
    min_fit = min(fitness)

    # 转换为“越大越好”的得分值（+1是防止除0）
    fitness_scores = [max_fit - f + 1e-6 for f in fitness]

    total_score = sum(fitness_scores)
    probabilities = [score / total_score for score in fitness_scores]

    # 构建累积概率
    cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

    # 轮盘赌选择
    new_population = []
    for _ in range(population_size):
        r = random.random()
        for i, cp in enumerate(cumulative_probabilities):
            if r <= cp:
                new_population.append(population[i])
                break


    #
    for i in population:
        flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(i, P, N, T,     setuptime)
        if not flag1:
            print("cuowu ")
            os._exit(1)
            try:
                sys.exit("程序终止：存在非法拓扑结构。")
            except SystemExit as e:
                raise e  # 强制抛出，不让 IDE 忽略

    return new_population

# 遗传算法主流程
def genetic_algorithm():
    ##1.我们先获得原始的卫星分布数据，主要是热点区域的拓扑序列数据


    ##2. 完成对热点区域预建链安排



    ## nodes是所个体的基础骨架
    # left_port 是所有个体随机生成的限制，可以认为，每个个体必须要满足同样的核心结构，就类似，每个个体可以长相不一样，基本上都要相同数量的骨头
    nodes = distinct_initial.distinct_initial(P, N, T, setuptime, regions_to_color)
    population =initialize_population(P,N,T,regions_to_color,setuptime)
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []

    writetoxml.nodes_to_xml(population[0], "E:\\code\\data\\2\\test_population.xml")










# 执行遗传算法
genetic_algorithm()
