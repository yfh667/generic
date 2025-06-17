import random
import math
import matplotlib.pyplot as plt
import random

import genaric2.writetoxml as writetoxml
import genaric2.initialize_individual as initialize_individual
import draw.snapshotf_romxml as snapshotf_romxml
import copy
import  genaric2.mutation.mutate as mutate
import genaric2.initiallink as initiallink
import genaric.chrom2adjact as c2a

import ga.graphalgorithm.mcmf.ssp_multi as ssp_multi
import genaric.plotgraph as plotgraph

import ga.graphalgorithm.adjact2weight as a2w
import os
import draw.snapshotf_romxml as snapshotf_romxml

# --- Example Usage ---
import genaric2.distinct_initial as distinct_initial
import genaric2.action_table as action_table
# 定义目标函数
import genaric2.cross.cross2 as cross
# import genaric2.cross.bfs_fitness as bfs_fitness
import genaric2.fitness.fitness as fitnessfuc
import genaric2.TopoSeqValidator as TopoSeqValidator

# 参数设置
#topology_prof





inter_link_bandwidth = 50
intra_link_bandwidth = 100

cost =1
population_size = 200  # 种群大小
generations = 500  # 最大迭代代数
mutation_rate = 0.3  # 变异概率
crossover_rate = 0.15  # 交叉概率

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








# 初始化种群
def initialize_population(P,N,T,nodes,setuptime):
    chromosome = []
    for i in range(population_size):
        nodes_copy = copy.deepcopy(nodes)  # 深复制nodes

        chromosome.append(   initialize_individual.initialize_individual(P, N, T, nodes_copy,setuptime))

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

    T = target_time_step
    setuptime = 2

    ## nodes是所个体的基础骨架
    # left_port 是所有个体随机生成的限制，可以认为，每个个体必须要满足同样的核心结构，就类似，每个个体可以长相不一样，基本上都要相同数量的骨头
    nodes = distinct_initial.distinct_initial(P, N, T, setuptime, regions_to_color)






    population =initialize_population(P,N,T,nodes,setuptime)
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []


    #
    # for i in population:
    #     flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(i, P, N, T, setuptime)
    #     if not flag1:
    #         print("initial cuowu  ")
    #         os._exit(1)
    #         try:
    #             sys.exit("程序终止：存在非法拓扑结构。")
    #         except SystemExit as e:
    #             raise e  # 强制抛出，不让 IDE 忽略




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
                child1, child2,point = cross.crossover(parent1,parent2, P, N, T, setuptime)
                next_population.append(child1)
                next_population.append(child2)




                flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(child1, P, N, T,
                                                                                                       setuptime)

                flag2, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(child2, P, N, T,
                                                                                                       setuptime)
                if  flag1==0 or flag2==0:
                    print(" zajiao cuowu ")
                    print(f"zajiao point{point} ")
                    writetoxml.nodes_to_xml(child1, "E:\\code\\data\\2\\child1.xml")
                    writetoxml.nodes_to_xml(child2, "E:\\code\\data\\2\\child2.xml")
                    writetoxml.nodes_to_xml(parent1, "E:\\code\\data\\2\\parent1.xml")
                    writetoxml.nodes_to_xml(parent2, "E:\\code\\data\\2\\parent2.xml")
                    os._exit(1)
                    try:
                        sys.exit("程序终止：存在非法拓扑结构。")
                    except SystemExit as e:
                        raise e  # 强制抛出，不让 IDE 忽略

                # try:
                #     # 尝试验证并保存数据
                #     flag1, conn1_test, conn2_test = TopoSeqValidator.TologialSequenceValidator(child1, P, N, T, setuptime)
                #     flag2, conn1_test, conn2_test = TopoSeqValidator.TologialSequenceValidator(child2, P, N, T, setuptime)
                #
                #     if flag1 == 0 or flag2 == 0:
                #         raise ValueError("非法拓扑结构")  # 主动抛出异常进入except块
                #
                # except Exception as e:  # 捕获所有可能的错误（包括验证器内部的报错）
                #     # 保存关键数据（确保即使验证崩溃也能保存）
                #     writetoxml.nodes_to_xml(child1, "E:\\code\\data\\2\\child1_debug.xml")
                #     writetoxml.nodes_to_xml(child2, "E:\\code\\data\\2\\child2_debug.xml")
                #     writetoxml.nodes_to_xml(parent1, "E:\\code\\data\\2\\parent1_debug.xml")
                #     writetoxml.nodes_to_xml(parent2, "E:\\code\\data\\2\\parent2_debug.xml")
                #
                #     # 打印错误信息
                #     print(f"【致命错误】{str(e)}", flush=True)
                #     print(f"交叉点: {point}", flush=True)
                #
                #     # 立即终止程序
                #     import os
                #     os._exit(1)  # 强制退出，避免任何可能的阻塞

        for i in next_population:
            flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(i, P, N, T,     setuptime)
            if  flag1==0:
                print(" cross cuowu ")
                os._exit(1)
                try:
                    sys.exit("程序终止：存在非法拓扑结构。")
                except SystemExit as e:
                    raise e  # 强制抛出，不让 IDE 忽略

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

            mutate_individual,mutate_node = mutate.mutate_f(individual, regions_to_color, inter_link_bandwidth, intra_link_bandwidth, cost, P, N,    T, setuptime)

            if mutate_individual!=individual:

                flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(mutate_individual, P, N, T,     setuptime)

                if flag1 == 0:
                    print("mutate cuowu ")
                    print(f"mutat node {mutate_node}")
                    writetoxml.nodes_to_xml(individual, "E:\\code\\data\\2\\para.xml")
                    writetoxml.nodes_to_xml(mutate_individual, "E:\\code\\data\\2\\child.xml")


                    try:
                        sys.exit("程序终止：存在非法拓扑结构。")
                    except SystemExit as e:
                        raise e  # 强制抛出，不让 IDE 忽略


                mutate_population.append(mutate_individual)






        for i in mutate_population:
            flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(i, P, N, T,     setuptime)
            if  flag1==0:
                print(" ,mutate cuowu ")
                os._exit(1)

                writetoxml.nodes_to_xml(child1, "E:\\code\\data\\2\\child1.xml")
                writetoxml.nodes_to_xml(child2, "E:\\code\\data\\2\\child2.xml")
                writetoxml.nodes_to_xml(parent1, "E:\\code\\data\\2\\parent1.xml")
                writetoxml.nodes_to_xml(parent2, "E:\\code\\data\\2\\parent2.xml")



                try:
                    sys.exit("程序终止：存在非法拓扑结构。")
                except SystemExit as e:
                    raise e  # 强制抛出，不让 IDE 忽略




        population = population + next_population+mutate_population

        population = selection(population,population_size,setuptime)


        print(f"populatetion nume={len(population)}")
        fitness, _ = fitnessfuc.fitness_function(population, regions_to_color, intra_link_bandwidth,  inter_link_bandwidth, cost, P, N, T)

        # fitness = bfs_fitness.fitness_function(P,N,T,population,regions_to_color)

        best_idx = fitness.index(min(fitness))
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_solution = population[best_idx]
         #   best_solution = decode_chromosome(P,N,T,population[best_idx])

        fitness_history.append(best_fitness)

        print(f"Generation {generation}: Best Fitness = {best_fitness:.4f},")
        fitness_sorted = sorted(fitness)

        print(f"Generation {generation}: all_fitness={fitness_sorted}")

    return best_solution, best_fitness, fitness_history


# 执行遗传算法
best_x, best_y, fitness_history = genetic_algorithm()
writetoxml.nodes_to_xml(best_x, "E:\\code\\data\\1\\best.xml")
print(f"Optimal solution: x = {best_x}")

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

# plotgraph.plot_graph_with_auto_curve_distinct(best_x,N,P,distinct)

