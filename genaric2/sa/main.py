
import random
import math
import matplotlib.pyplot as plt
import random

import genaric2.writetoxml as writetoxml
import genaric2.initialize_individual as initialize_individual
import genaric2.TopoSeqValidator as TopoSeqValidator
import  genaric2.mutation3_probability.mutate as mutate
import  genaric2.mutation.mutate as mutate_raw

import os
import draw.snapshotf_romxml as snapshotf_romxml



import genaric2.fitness.fitness as fitnessfuc
import draw.snapshotf_romxml as snapshotf_romxml
import graph.drawall as drawall
import genaric2.tegnode as tegnode

import graph.time_2d3 as time_2d
# --- Example Usage ---
import genaric2.distinct_initial as distinct_initial
import genaric2.action_table as action_table
import copy


import genaric2.writetoxml as writetoxml
import genaric2.initialize_individual as initialize_individual


# 参数设置
#topology_prof
#this function ,based on generic2.
inter_link_bandwidth = 50
intra_link_bandwidth = 100

cost =1





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




import numpy as np
def test_function(x):
    return np.sin(x) + np.sin(10 * x / 3)







# Generate a neighboring solution
def generate_neighbor(solution,):
    #perturbation = np.random.uniform(-step_size, step_size)
    #candidate = solution + perturbation
    # limit new solutions to the search space
   # candidate = np.clip(candidate, min_max[0], min_max[1])


    candidate, mutate_node,_ ,_= mutate.mutate_f(solution, regions_to_color, inter_link_bandwidth, intra_link_bandwidth, cost, P, N, T, setuptime)

    return candidate,mutate_node

# Calculate acceptance probability
def acceptance_probability(current_cost, new_cost, temperature):


    if new_cost[0] < current_cost[0]:
        return 1.0
    else:
        return np.exp(-(new_cost[0] - current_cost[0]) / (1e-8 + temperature))

# Simulated Annealing main.py function
def simulated_annealing(initial_solution, objective_function, T_start, alpha, num_iterations):
    current_solution = initial_solution


    indictor, seq = objective_function(current_solution,regions_to_color,inter_link_bandwidth,intra_link_bandwidth,cost,P, N, T)

    current_cost= [indictor, seq]

    best_solution = current_solution
    best_cost = current_cost
    accepted_solutions = [current_solution]
    costs = [current_cost]

    temperature = T_start
    for iteration in range(num_iterations):
        neighbor,_ = generate_neighbor(current_solution,)

        flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(
            neighbor, P, N, T,regions_to_color, setuptime)
        if flag1 == 0:
            # print(f"mutate node {mutate_node},neighbor ={chose}")
            # writetoxml.nodes_to_xml(individual, "E:\\code\\data\\2\\para.xml")
            print("0")
            writetoxml.nodes_to_xml(neighbor, "E:\\code\\data\\2\\sa_debug.xml")
# here


        indictor, seq = objective_function(neighbor,regions_to_color,inter_link_bandwidth,intra_link_bandwidth,cost,P, N, T)
        neighbor_cost=[indictor, seq]
        if acceptance_probability(current_cost, neighbor_cost, temperature) > np.random.random():
            current_solution = neighbor
            current_cost = neighbor_cost
            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost





        accepted_solutions.append(current_solution)
        costs.append(current_cost)
        temperature *= alpha

    return accepted_solutions, best_solution, best_cost, costs


if __name__ == "__main__":
    # Run the algorithm

    # Run the algorithm
    T_start = 3000
    alpha = 0.99
    num_iterations = 500

  #  initial_solution = np.random.uniform(min_max[0], min_max[1])



    T = target_time_step
    setuptime = 2
    initial_solution=initialize_individual.initialize_individual_region(regions_to_color, P, N, T, setuptime)






    results = simulated_annealing(initial_solution,fitnessfuc.calculate_fitness_onedividual , T_start, alpha, num_iterations)

    # Plot the results
    accepted_solutions, best_solution, best_cost, costs = results
    writetoxml.nodes_to_xml(best_solution, "E:\\code\\data\\1\\sa_best.xml")


   # print("accepted solutions:", accepted_solutions)
   # print("best solution:", best_solution)
    print("best cost:", best_cost)
    #print("costs:", costs)


    allcost = []

    for i in costs:
        allcost.append(i[0])
    # Plot the results

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(allcost, label='Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost by Iteration')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Output the best solution found
    best_cost

    adjacency_list_array=action_table.action_map2_shanpshots(best_solution, P, N, T)
    ########
    flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(best_solution, P, N, T,regions_to_color,
                                                                                           setuptime)

    weight_list_array = []

    for t in range(T):
        weight = {}
        for i in range(P):
            for j in range(N):
                key = i * N + j
                value = best_solution[(i, j, t)].importance
                weight[key] = value
        weight_list_array.append(weight)



    vis = time_2d.DynamicGraphVisualizer(
            adjacency_list_array,
            regions_to_color,
            N,
            P,
            node_weights_array=weight_list_array  # ← 这一行传入节点权重
    )
    vis.show(block=True)
