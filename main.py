import random
import numpy as np
import matplotlib.pyplot as plt
#from pyantcolony import AntColony

# Number of jobs and machines
jobs = 3
machines = 2

# Processing time of each job on each machine
processing_time = np.random.randint(1, 10, (jobs, machines))

# Initial population
population = np.random.randint(0, 2, (100, jobs*machines))

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
plot_gantt(best_individual, processing_time)