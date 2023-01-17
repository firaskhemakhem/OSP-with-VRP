import random
import numpy as np
import matplotlib.pyplot as plt
import time
from random import randint
import plotly.graph_objects as go

# Number of jobs and machines
jobs = 5
machines = 3

# Processing time of each job on each machine
processing_time = np.random.randint(1, 10, (jobs, machines))

# Initial population
population = np.random.randint(0, 2, (100, jobs*machines))
best_schedule = []
# Fitness function
def fitness(individual):
    completion_time = np.zeros(machines)
    for job in range(1,jobs):
        for machine in range(1,machines):
            if individual[job*machines + machine] == 1:
                completion_time[machine] += processing_time[job][machine]
    return max(completion_time)

# Selection function
def selection(population, fitnesses):
    parents = np.zeros((2, len(population[0])))
    for i in range(2):
        max_fitness = 0
        max_index = 0
        for j in range(len(population)):
            if fitnesses[j] > max_fitness:
                max_fitness = fitnesses[j]
                max_index = j
        parents[i] = population[max_index]
        fitnesses[max_index] = 0
    return parents

# Crossover function
def crossover(parents):
    crossover_point = np.random.randint(0, len(parents[0]))
    child = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
    return child

# Mutation function
def mutation(child):
    mutation_point = np.random.randint(0, len(child))
    child[mutation_point] = 1 - child[mutation_point]
    return child

# Main loop
for generation in range(1000):
    fitnesses = [fitness(individual) for individual in population]
    parents = selection(population, fitnesses)
    child = crossover(parents)
    child = mutation(child)
    population[np.argmin(fitnesses)] = child

# Best individual
best_individual = population[np.argmin(fitnesses)]
best_fitness = min(fitnesses)
print(best_individual)
# Function to plot Gantt chart
def plot_gantt(best_individual, processing_time):
    completion_time = np.zeros(machines)
    schedule = []
    for job in range(jobs):
        for machine in range(machines):
            if best_individual[job*machines + machine] == 1:
                schedule.append([job, machine, completion_time[machine], completion_time[machine]+processing_time[job][machine]])
                completion_time[machine] += processing_time[job][machine]
    schedule.sort(key=lambda x: x[2])
    print(schedule)
    plt.figure(figsize=(20, 110))
    for i in range(len(schedule)):
        plt.barh(schedule[i][1], schedule[i][3]-schedule[i][2], left=schedule[i][2], height=0.6, label='Job '+str(schedule[i][0]))
        #plt.broken_barh([(schedule[i][2],schedule[i][2])],(schedule[i][1],1),facecolors='blue')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Machines")
    plt.show()
    return schedule
best_schedule = plot_gantt(best_individual, processing_time)

osp_solution,osp_fitness = best_individual,best_fitness

### VRP
def VRP_sol(n_jobs, n_veh, capacity, jobs_demand, distance_matrix, max_time = 10):
    start_time = time.time()
    # Initial solution : assign each job to a random vehicle
    current_solution = [[] for _ in range(n_veh)]
    for i, demand in enumerate(jobs_demand):
        current_solution[random.randint(0, n_veh-1)].append(i)
    current_cost = VRP_cost(current_solution, capacity, jobs_demand, distance_matrix)
    best_solution = current_solution
    best_cost = current_cost
    while True:
        # Neighbors generation :
        # 1. Remove a job from a vehicle and add it to another one
        # 2. Swap the position of 2 jobs in the same vehicle
        neighbor_solution = [route[:] for route in current_solution]
        if random.random() < 0.5:
            # Remove a job from a vehicle and add it to another one
            vehicle = random.randint(0, n_veh-1)
            if len(neighbor_solution[vehicle]) > 0:
                job = neighbor_solution[vehicle].pop(random.randint(0, len(neighbor_solution[vehicle])-1))
                vehicle_dest = random.randint(0, n_veh-1)
                while vehicle == vehicle_dest:
                    vehicle_dest = random.randint(0, n_veh-1)
                neighbor_solution[vehicle_dest].append(job)
        else:
            # Swap the position of 2 jobs in the same vehicle
            vehicle = random.randint(0, n_veh-1)
            if len(neighbor_solution[vehicle]) > 1:
                job1, job2 = random.sample(neighbor_solution[vehicle], 2)
                neighbor_solution[vehicle][neighbor_solution[vehicle].index(job1)], neighbor_solution[vehicle][neighbor_solution[vehicle].index(job2)] = job2, job1
        # Acceptance criterion
        neighbor_cost = VRP_cost(neighbor_solution, capacity, jobs_demand, distance_matrix)
        if neighbor_cost < current_cost or random.random() < 0.1:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
        # Stop condition
        if time.time() - start_time > max_time:
            break
    return best_solution

def VRP_cost(solution, capacity, jobs_demand, distance_matrix):
    cost = 0
    for route in solution:
        load = 0
        for i in range(len(route)):
            load += jobs_demand[route[i]]
            if i < len(route)-1:
                cost += distance_matrix[route[i]][route[i+1]]
            if load > capacity:
                return float('inf')
        if route:
          cost += distance_matrix[route[0]][route[-1]]
    return cost



def plot_gantt_vrp(vrp_solution, distance_matrix):
    plt.figure(figsize=(10, 5))
    current_time = 0
    current_position = 0
    n_veh = 0
    for i, vehicle_route in enumerate(vrp_solution):
        for j, job in enumerate(vehicle_route):
            # Add the task to the plot
            plt.barh(i, distance_matrix[current_position][job], height=0.6, label='Job {}'.format(job))
            # Update the current position and the current time
            current_position = job
            current_time += distance_matrix[current_position][job]

        if len(vehicle_route) != 0:
            # Add the return task to the plot
            #plt.barh(i, distance_matrix[current_position][0], left=current_time, height=0.6, label='Return')
            current_time += distance_matrix[current_position][0]
        # Add the vehicle to the plot
        n_veh += 1
    plt.yticks(range(n_veh))
    plt.xlabel('Time')
    plt.legend()
    plt.show()
# Example of usage

complited_jobs = []
for i in best_schedule:
    complited_jobs.append(i[0])
complited_jobs = list(set(complited_jobs))
n_jobs = len(complited_jobs)
n_veh = 3
capacity = 40
jobs_demand = [random.randint(1, 10) for _ in range(n_jobs)]
distance_matrix = [[random.randint(1, 10) for _ in range(n_jobs)] for _ in range(n_jobs)]
best_sol = VRP_sol(n_jobs, n_veh, capacity, jobs_demand, distance_matrix)
print(best_sol)

plot_gantt_vrp(best_sol, distance_matrix)
