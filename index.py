import random
import numpy as np
import matplotlib.pyplot as plt


X = np.concatenate([np.random.randn(50, 2), 5+ np.random.randn(50, 2), 10 + np.random.randn(50, 2)])


def fitness_function(population):
    labels = np.zeros(len(X))
    for i, x in enumerate(X):
        distances = np.sqrt(np.sum((population - x)**2, axis=-1))
        labels[i] = np.argmin(distances)
    score = -len(np.unique(labels))
    return score

POPULATION_SIZE = 50
ELITE_SIZE = 5
MUTATION_RATE = 0.1
GENERATIONS = 50

population = np.random.rand(POPULATION_SIZE, 2) * 10

# Run Genetic Algorithm
fitness_history = []
for generation in range(GENERATIONS):
    # Evaluate fitness
    fitness_scores = [fitness_function(individual) for individual in population]
    fitness_history.append(np.max(fitness_scores))

    parents = population[np.argsort(fitness_scores)[-ELITE_SIZE:]]
    
    offspring = []
    for i in range(POPULATION_SIZE - ELITE_SIZE):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = np.mean([parent1, parent2], axis=0)
        offspring.append(child)
    offspring = np.array(offspring)
    
    for i in range(len(offspring)):
        if np.random.rand() < MUTATION_RATE:
            offspring[i] += np.random.randn(2)


    population = np.concatenate((parents, offspring))
 


labels = np.zeros(len(X))
for i, x in enumerate(X):
    distances = np.sqrt(np.sum((population - x)**2, axis=-1)) # Fix axis error
    labels[i] = np.argmin(distances)
plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("Cluster Analysis")
plt.show()