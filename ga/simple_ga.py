import random
import math
import matplotlib.pyplot as plt
import random
import numpy as np
from networkx.classes import neighbors

import graph_ga.chrom2adjact as c2a
import graph_ga.ga.initiallink as initiallink
# 示例用法
import graph_ga.adjact2chrom as a2c
import graph_ga.dijstra as dij
import graph_ga.chrom2adjact as c2a
import graph_ga.plotgraph as plotgraph
# 定义目标函数

# 参数设置
#topology_prof
N=4
P=4

population_size = 100  # 种群大小
generations = 100  # 最大迭代代数
mutation_rate = 0.2  # 变异概率
crossover_rate = 0.25  # 交叉概率
chromosome_length = N*(P-1)  # 二进制染色体长度（22位）

#
# adjacency_example = {
#     1: [(2, 36), (3, 27)],
#     2: [(3, 13), (4, 18), (5, 20)],
#     3: [(5, 12), (6, 23)],
#     4: [(7, 11), (8, 32)],
#     5: [(4, 16), (6, 30)],
#     6: [(7, 12), (9, 38)],
#     7: [(8, 20), (9, 32)],
#     8: [(9, 15), (10, 24)],
#     9: [(10, 13)],
#     10: [],
#     # 其他没列出的节点则不连线
# }


# 将二进制染色体转换为十进制数值
def decode_chromosome(chromosome):


    base_adjacency_list = c2a.base_chrom2adjacent(chromosome, N, P)

    adjacency_list = c2a.full_adjacency_list(base_adjacency_list, N, P)


    return adjacency_list
def initialize_individual(N,P):
    individual = [-1]*(P-1)*N
    #left_link =[0]*(P)*N
    for i in range(P-1):
        nowlink = initiallink.initialize_links(N)
        for j in range(len(nowlink)):
         #   individual[j] = nowlink[j]
            nowlink[j] = (i + 1) * N +  (nowlink[j] % N)  #

        individual[i*N:(i+1)*N]=nowlink

    return individual

# 初始化种群
def initialize_population():
    chromosome = []
    for i in range(population_size):
        chromosome.append ( initialize_individual(N,P))

    return chromosome

   # return [random.sample(range(chromosome_length), chromosome_length) for _ in range(population_size)]


# 计算种群适应度
def fitness_function(population):
    indictors = []
    decoded_values = [decode_chromosome(ind) for ind in population]
    for adjacency_list in decoded_values:
        start = 0
        end = 14

        indictor = 0
        path, distance = dij.dijkstra_shortest_path(adjacency_list, start, end)
        #    print(f"最短路径: {path}")  # 输出: [0, 3, 6, 7, 8]
        #  print(f"最短距离: {distance}")  # 输出: 4
        indictor = indictor + distance

        start = 3
        end = 13
        path, distance = dij.dijkstra_shortest_path(adjacency_list, start, end)
        indictor = indictor + distance
        indictors.append(indictor)
    #  print(indictor)

    # Rastrigin函数（多峰函数，常用于测试优化算法）

    return indictors

    # allvalue = []
    # decoded_values = [decode_chromosome(ind) for ind in population]
    # for ind in range(len(decoded_values)):
    #     # decoded_values[ind] represeent the path
    #     value = 0
    #     path = decoded_values[ind]
    #
    #     for nodeid in range(len(path) - 1):
    #
    #         start = path[nodeid]
    #         end = path[nodeid + 1]
    #         neighbors = adjacency_example[start]
    #         for neighbor in neighbors:
    #             nodeid = neighbor[0]
    #             if (nodeid == end):
    #                 value = value + neighbor[1]
    #                 break
    #     value = 1 / value
    #     allvalue.append(value)
    #
    # return allvalue


# 选择操作（轮盘赌选择）
def selection(population):
    fitness = fitness_function(population)
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


	# 交叉操作
# 这里，我们要注意，可能
def crossover_wenti(child1):
    # 1. 去重并保留原顺序
    # unique_child = []
    seen = set()
    chonfguid = []
    for i in range(len(child1)):
        if child1[i] not in seen:

            # unique_child.append(num)
            seen.add(child1[i])
        else:
            chonfguid.append(i)

    # here we get the chongfu id
    # 2. 找到缺失的数字
    full_set = set(range(1, len(child1) + 1))
    missing_numbers = list(full_set - seen)

    # 3. 补足缺失数字

    for i in range(len(chonfguid)):
        child1[chonfguid[i]] = missing_numbers[i]

    return child1


def crossover(parent1, parent2):

    point = random.randint(1, P - 1)
    child1 = parent1[:point*N] + parent2[point*N:]
    child2 = parent2[:point*N] + parent1[point*N:]
    return child1, child2

    # 这里，我们开始进行处理冲突点
   # return crossover_wenti(child1), crossover_wenti(child2)


# 变异操作
def mutate(individual):
    individual_copy = individual.copy()  # 创建副本
    #random_index = random.sample(range(len(individual_copy)), 1)
    random_index = random.choice(range(len(individual_copy)))

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
    population = initialize_population()
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []

    for generation in range(generations):

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
                child1, child2 = crossover(parent1, parent2)

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
            mutate_individual = mutate(individual)

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

plotgraph.plot_graph_with_auto_curve(best_x,N,P)

