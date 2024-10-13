import random

from mpmath import extend


# code from https://www.youtube.com/watch?v=CRtZ-APJEKI

population_Size = 100
genome_length = 20
mutation_rate = .02
crossover_rate = 0.70
generations = 2000

def randGenome(length):
    return[random.randint(0,1) for _ in range(length)]

def init_population(population_size, genome_length):
    return [randGenome(genome_length) for _ in range(population_size)]


def fitness(genome):
   return sum(genome)

def selectParent(population, fitness_vals):
    total_fitness = sum(fitness_vals)
    pick = random.uniform(0,total_fitness)
    current = 0
    for individual, fitness_val in zip(population, fitness_vals):
        current += fitness_val
        if current > pick:
            return individual

def crossover(p1,p2):
    if random.random() < crossover_rate:
        cross_point = random.randint(1, len(p1)-1)
        return p1[:cross_point] + p2[cross_point:], p2[:cross_point] + p1[cross_point:]
    else:
        return p1, p2

def mutate(genome):
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome[i] = abs(genome[i]-1)
    return genome

def genetic_algo():
    population = init_population(population_Size,genome_length)

    for generation in range(generations):
        fitness_values = [fitness(genome) for genome in population]

        new_pop = []
        for _ in range(population_Size // 2):
            p1 = selectParent(population, fitness_values)
            p2 = selectParent(population, fitness_values)
            baby1, baby2 = crossover(p1,p2)
            new_pop.extend([mutate(baby1), mutate(baby2)])

        population = new_pop

        fitness_values = [fitness(genome) for genome in population]
        best_fitness = max(fitness_values)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    best_index = fitness_values.index(max(fitness_values))
    best_solution = population[best_index]
    print(f' Best Solution: {best_solution}')
    print(f'Best Fitness: {fitness(best_solution)}')


if __name__ == '__main__':
    genetic_algo()
