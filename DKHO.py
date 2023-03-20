import copy
import numpy
import multiprocessing
import matplotlib.pyplot as plt
import cube
import random
from deap import base, algorithms
from deap import creator
from deap import tools


class DKHO:
    hall_of_fame = []
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Particle", list, fitness=creator.Fitness, best=None)
    creator.create("Swarm", list, gbest=None, gbestfit=creator.Fitness)

    def __init__(self, cube_to_solve, NUM_KRILL, NGEN, CXPB, MUTPB, EVAL_DEPTH, MIN_MUTATE, MAX_MUTATE, SELECTION_SIZE, PARSIMONY_SIZE, LAMBDA):
        self.NUM_KRILL = NUM_KRILL
        self.NGEN = NGEN
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.EVAL_DEPTH = EVAL_DEPTH
        self.MIN_MUTATE = MIN_MUTATE
        self.MAX_MUTATE = MAX_MUTATE
        self.SELECTION_SIZE = SELECTION_SIZE
        self.PARSIMONY_SIZE = PARSIMONY_SIZE
        self.shuffled_cube = cube_to_solve
        self.LAMBDA = LAMBDA
        pool = multiprocessing.Pool()

        toolbox = base.Toolbox()
        toolbox.register("map", pool.map)
        toolbox.register("particle", self.init_krill, creator.Particle)
        toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
        toolbox.register("mate", self.safe_cxOnePoint)
        toolbox.register("mutate", self.mutate, min_mutate=self.MIN_MUTATE, max_mutate=self.MAX_MUTATE,
                         indpb=self.MUTPB)
        toolbox.register("select", tools.selDoubleTournament, fitness_size=self.SELECTION_SIZE,
                         parsimony_size=self.PARSIMONY_SIZE,
                         fitness_first=True, )
        toolbox.register("evaluate", self.fitness, self.EVAL_DEPTH)

        swarm = toolbox.swarm(n=self.NUM_KRILL)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)
        logbook = tools.Logbook()
        hof = tools.HallOfFame(10)

        swarm, logbook = self.eaMuPlusLambdaWithMoveSelection(swarm, toolbox, self.NUM_KRILL, self.LAMBDA, self.CXPB,
                                                              self.MUTPB, self.NGEN, stats,
                                                              hof,
                                                              verbose=True)

        hof.update(swarm)
        self.logbook = logbook
        self.hall_of_fame = hof

    def fitness(self, depth, krill):
        """
        Assesses the fitness of a krill by getting the length of the solution generated by Kociemba's algorithm on the cube
        state that the krill is currently in.
        :param depth: Depth to run Kociemba's algorithm
        :param krill: Krill individual to assess
        :return: A tuple of the length of the solution found
        """
        temp_cube = copy.deepcopy(self.shuffled_cube)
        temp_cube.run_moves(krill)
        solution = []
        for i in range(depth):
            solve = temp_cube.solve_kociemba()
            if len(solve) < len(solution):
                solution = solve
            elif not solution:
                solution = solve

        if not krill.best:
           krill.best = krill
        elif krill.fitness.values[0] < krill.best.fitness.values[0]:
            krill.best = krill

        if solution == ['']:
            return 0,
        else:
            return len(solution),

    def mutate(self, krill, min_mutate=0, max_mutate=0, indpb=0.0):
        """
        To mutate the krill, we randomly change some of its moves. The number of moves changed are between min_mutate and
        max_mutate. indpb is the probability of a krill being mutated
        :param krill: Krill individual to be possibly mutated
        :param min_mutate: Minimum number of mutations
        :param max_mutate: Maximum number of mutations
        :param indpb: Probability of a Krill being mutated
        :return: The (possibly) mutated krill
        """
        if indpb > random.random():
            if len(krill) == 0:
                return krill,
            elif len(krill) == 1:
                mutates = [0]
            elif len(krill) < max_mutate:
                mutates = random.sample(range(0, len(krill)), k=random.randint(min_mutate, len(krill)))
            else:
                try:
                    mutates = random.sample(range(0, len(krill)), k=random.randint(min_mutate, max_mutate))
                except ValueError:
                    print("Somethings gone wrong with mutation!")
                    raise ValueError

            for index in mutates:
                krill[index] = self.shuffled_cube.random_moves(1)[0]
        return krill,

    def move_selection(self, swarm):
        for krill in swarm:
            threshold = krill.fitness
            fitnesses = {}

            # Current state the krill is in is krill_cube
            krill_cube = copy.deepcopy(self.shuffled_cube)
            krill_cube.run_moves(krill)


            for move in cube.Cube.move_map.keys():
                # We will clone this cube and assess each possible move on it
                temp_cube = copy.deepcopy(krill_cube)
                temp_cube.run_moves([move])
                fitnesses[move] = len(temp_cube.solve_kociemba())

            # We now have a dictionary of the fitnesses of each move

            # We will get the best move(s)
            minval = min(fitnesses.values())
            best_moves = list(filter(lambda x: fitnesses[x] == minval, fitnesses))

            # If the best move(s) have a fitness below the threshold, we will choose it (or a random one if multiple)
            if minval < threshold.values[0]:
                if len(best_moves) > 1:
                    chosen_move = random.choice(best_moves)
                else:
                    chosen_move = best_moves[0]
            # Otherwise we will randomly choose a move based on its fitness
            else:
                # We also give the option for the krill to not move
                fitnesses[''] = threshold.values[0]
                chosen_move = self.weighted_random_choice(fitnesses)

            krill.append(chosen_move)
            try:
                krill.remove('')
            except ValueError:
                return

    def safe_cxOnePoint(self, krill1, krill2):
        if len(krill1) <=1 or len(krill2) <= 1:
            return krill1, krill2
        else:
            return tools.cxOnePoint(krill1,krill2)


    def weighted_random_choice(self, choices):
        max = sum(choices[choice] for choice in choices)
        pick = random.uniform(0, max)
        current = 0
        for choice in choices:
            current += choices[choice]
            if current > pick:
                return choice

    def init_krill(self, krill):
        krill = creator.Particle()
        return krill

    # This is taken from the DEAP GitHub, I've replicated it here so I can modify it to have move selection for the krill
    def eaMuPlusLambdaWithMoveSelection(self, population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                                        stats=None, halloffame=None, verbose=__debug__):
        r"""This is the :math:`(\mu + \lambda)` evolutionary algorithm.
        :param population: A list of individuals.
        :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                        operators.
        :param mu: The number of individuals to select for the next generation.
        :param lambda_: The number of children to produce at each generation.
        :param cxpb: The probability that an offspring is produced by crossover.
        :param mutpb: The probability that an offspring is produced by mutation.
        :param ngen: The number of generation.
        :param stats: A :class:`~deap.tools.Statistics` object that is updated
                      inplace, optional.
        :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                           contain the best individuals, optional.
        :param verbose: Whether or not to log the statistics.
        :returns: The final population
        :returns: A class:`~deap.tools.Logbook` with the statistics of the
                  evolution.
        The algorithm takes in a population and evolves it in place using the
        :func:`varOr` function. It returns the optimized population and a
        :class:`~deap.tools.Logbook` with the statistics of the evolution. The
        logbook will contain the generation number, the number of evaluations for
        each generation and the statistics if a :class:`~deap.tools.Statistics` is
        given as argument. The *cxpb* and *mutpb* arguments are passed to the
        :func:`varOr` function. The pseudocode goes as follow ::
            evaluate(population)
            for g in range(ngen):
                offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
                evaluate(offspring)
                population = select(population + offspring, mu)
        First, the individuals having an invalid fitness are evaluated. Second,
        the evolutionary loop begins by producing *lambda_* offspring from the
        population, the offspring are generated by the :func:`varOr` function. The
        offspring are then evaluated and the next generation population is
        selected from both the offspring **and** the population. Finally, when
        *ngen* generations are done, the algorithm returns a tuple with the final
        population and a :class:`~deap.tools.Logbook` of the evolution.
        This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
        :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
        registered in the toolbox. This algorithm uses the :func:`varOr`
        variation.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        count_gen = 0
        solution_found = False

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        #for gen in range(1, ngen + 1):
        while count_gen < ngen and not solution_found:
            count_gen += 1
            # Let each krill make a move
            self.move_selection(population)

            # Vary the population
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                if not population.gbest:
                    population.gbest = ind
                elif ind.fitness.values[0] < population.gbest.fitness.values[0]:
                    population.gbest = ind
                if ind.fitness.values[0] == 0:
                    solution_found = True

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(offspring, mu)

            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=count_gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

    def get_hof(self):
        return self.hall_of_fame

    def get_logbook(self):
        return self.logbook


if __name__ == '__main__':
    shuffled_cube = cube.Cube(3)
    shuffle_moves = shuffled_cube.random_moves(5)
    shuffled_cube.run_moves(shuffle_moves)
    DKHO(shuffled_cube)
