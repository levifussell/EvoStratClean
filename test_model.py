import numpy as np
import pickle as pkl
from sys import argv

from utils import individual_to_neural_network
from evolution_objective import EvolutionObjective
from alg_es import EsOptimiser

import gym


def run_test(base, VIRTUAL_BATCH):
    '''
    Given a pre-trained model this will test it on the environment.
    '''

    env = gym.make('BipedalWalker-v2')
    STATE_SIZE = np.shape(env.observation_space)[0]
    ACTION_SIZE = np.shape(env.action_space)[0]

    LAYER_SIZES = [STATE_SIZE, 64, ACTION_SIZE]
    LAYER_ACTIVATIONS = ['tanh', 'tanh']
    BIASES = True
    WEIGHTS_TOTAL = 0
    for l1,l2 in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:]):
        WEIGHTS_TOTAL += l1*l2
    BIASES_TOTAL = np.sum(LAYER_SIZES[1:])
    VBN = True
    VBN_TOTAL = len(LAYER_SIZES[1:-1])*2

    obs = env.reset()
    reward_total = 0.0

    def to_nn(individual):
        return individual_to_neural_network(individual=individual, layer_sizes=LAYER_SIZES, layer_activations=LAYER_ACTIVATIONS, biases=BIASES,
                                            use_vbn=VBN, virtual_batch=VIRTUAL_BATCH) 

    env = EvolutionObjective(environment=env, repetitions=1, to_model=to_nn) 

    fitness = env.compute(base, no_terminate=False)
    print("FITNESS: {}".format(fitness))

if __name__ == "__main__":
    filename = argv[1]
    base, virtual_batch = pkl.load(open(filename, "rb"))
    print(base)
    print(virtual_batch)
    run_test(base, virtual_batch)
