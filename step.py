from numpy.random import rand, randint
import numpy as np
import matplotlib.pyplot as plt


def objective_function(I):
    x1 = I[0]
    x2 = I[1]
    objective_min = (x1**2+x2**2-x1*x2-10)*10**-3
    objective_max = 1/(1 + objective_min) # Convert the min to max problem

    return objective_max

# Parameters of the binary genetic algorithm
bounds = [[-50,50], [-50,50]]
iteration = 100
bits = 20 #number of bits for each variable
pop_size = 50
crossover_rate = 0.8
mutation_rate = 0.2

# 
#  the rest of the python code can be kept the same
def crossover(pop, crossover_rate):
    offspring = list()
    for i in range(int(len(pop)/2)):
        p1 = pop[2*i-1].copy() #parent 1 
        p2 = pop[2*i].copy() #parent 2
        if rand() < crossover_rate:
            cp = randint(1, len(p1)-1,size=2) # two random cutting points
            while cp[0]==cp[1]:
                cp=randint(1, len(p1)-1,size=2) #two random cutting points
            
            cp = sorted(cp)
            c1 = p1[:cp[0]] + p2[cp[0]:cp[1]] + p1[cp[1]:]
            c2 = p2[:cp[0]] + p1[cp[0]:cp[1]] + p1[cp[1]:]
            offspring.append(c1)
            offspring.append(c2)
        else:
            offspring.append(p1)
            offspring.append(p2)

    return offspring

print(crossover)
print(objective_function)