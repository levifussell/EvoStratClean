import numpy as np
import pickle as pkl
import time

from utils import OptAdam, OptSgd

class EsOptimiser():

    def __init__(self, population_size=100, noise_size=0.001, dimension_size=100, step_size=0.001,
                    select_percentage=0.8, virtual_batch=None, #heuristic_rate=1.01, 
                    init_base_function=None):
        self.population_size = population_size
        self.noise_size = noise_size
        self.dimension_size = dimension_size
        self.step_size = step_size
        self.select_percentage = select_percentage
        self.virtual_batch = virtual_batch
        self.prev_f_base = -100000000

        # optimiser
        self.opt = OptAdam(step_size=step_size, wind_mom_rate=0.9, wind_grad_rate=0.999)
        #self.opt = OptSgd(step_size=step_size)

        # create the base individual
        if init_base_function is not None:
           self.base = init_base_function(dimension_size)
        else:
           self.base = np.random.randn(1, dimension_size)

        # initialise utility values
        self.utilities = np.zeros(self.population_size)
        for p in range(self.population_size):
            self.utilities[p] = np.max([0.0, np.log(float(self.population_size)/2.0 + 1.0) - np.log(p+1.0)])

        # normalise
        self.utilities /= np.sum(self.utilities)
        self.utilities -= 1.0/float(self.population_size)

    def get_individual_noise(self, seed, ind_mult=1.0):
        np.random.seed(seed)
        return np.random.randn(*np.shape(self.base))*ind_mult

    def get_individual(self, seed, ind_mult=1.0):
        """
        for memory efficiency, individuals are stored as random seeds
        ind_mult : is used to swap between opposite individuals
        """
        ind = self.base + self.get_individual_noise(seed=seed, ind_mult=ind_mult)*self.noise_size
        return ind

    def print_progress(self, episode, avg_fitness, best_fitness, std_fitness):
        print("EP: {} - AVG-F: {}, BEST-F: {}, STD-F: {}".format(episode, avg_fitness, best_fitness, std_fitness))

    def step(self, objective):

        half_pop = int(self.population_size/2)
        fitnesses = np.zeros(self.population_size)
        ind_muls = np.ones(self.population_size)
        ind_muls[half_pop:] *= -1.0

        # first compute the fitness of each front
        #  seed of the individual is just its index
        #print("VBN: {}".format(self.base[:, -2:]))
        np.random.seed(int(time.time()))
        seed_offset = np.random.randint(1000000)

        f_base = objective.compute(self.base)
        self.prev_f_base = f_base

        for p in range(self.population_size):
            ind = self.get_individual(seed=seed_offset + (p % half_pop), ind_mult=ind_muls[p])
            f = objective.compute(ind)
            fitnesses[p] = f
            
        # uncomment to print the best individual each epoch.
        #print("ind 1: {}".format(self.get_individual_noise(seed=seed_offset)))

        # sorting negative means it does high to low
        sort_idx = np.argsort(-fitnesses) 

        # compute stats for logging
        f_best = np.max(fitnesses)
        f_mean = np.mean(fitnesses)
        f_std = np.std(fitnesses)

        # add a step in the direction proportional to the fitness
        step = np.zeros_like(self.base)
        for p in range(self.population_size):
            step += self.utilities[p] * self.get_individual_noise(seed=seed_offset + (sort_idx[p] % half_pop), ind_mult=ind_muls[sort_idx[p]])

        # normalise the step size
        step /= float(self.noise_size)


        # do the step
        self.base += self.opt.step(gradient=step)
        # uncomment to print the gradient step each epoch
        #print("step: {}".format(step*self.step_size))

        return f_mean, f_best, f_std, self.get_individual(seed=seed_offset + (sort_idx[0] % half_pop), ind_mult=ind_muls[sort_idx[p]])

    def run_steps(self, objective, num_runs=1000, print_out=True, run_name="TEST"):

        f_best_base = -100000000
        f_best_ind = -100000000
        for r in range(num_runs):
            f_m, f_b, f_s, best_ind = self.step(objective)
            
            if print_out:
                self.print_progress(episode=r, avg_fitness=f_m, best_fitness=f_b, std_fitness=f_s)

            if f_m > f_best_base:
                # save base
                f_best_base = f_m
                self.save_base(run_name+"_MODEL.pkl")

                # save best individual
            if f_b > f_best_ind:
                f_best_ind = f_b
                self.save_best(best_ind, run_name+"MODEL_best_ind.pkl")

    def save_base(self, filename):
        if self.virtual_batch is not None:
            pkl.dump((self.base, self.virtual_batch), open(filename, "wb"))
            print("SAVED BASE (w/ virtual batch)")
        else:
            pkl.dump(self.base, open(filename, "wb"))
            print("SAVED BASE")

    def save_best(self, best_ind, filename):
        if self.virtual_batch is not None:
            pkl.dump((best_ind, self.virtual_batch), open(filename, "wb"))
            print("SAVED IND (w/ virtual batch)")
        else:
            pkl.dump(best_ind, open(filename, "wb"))
            print("SAVED IND")

