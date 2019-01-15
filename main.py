import numpy as np
from sys import argv

import argparse

from utils import individual_to_neural_network #, mirror_action
from evolution_objective import EvolutionObjective
from alg_es import EsOptimiser

import gym

env = gym.make('BipedalWalker-v2') # TODO

STATE_SIZE = np.shape(env.observation_space)[0]
ACTION_SIZE = np.shape(env.action_space)[0]

LAYER_SIZES = [STATE_SIZE, 64, ACTION_SIZE]
LAYER_ACTIVATIONS = ['tanh', 'tanh']
BIASES = True
VBN = True
WEIGHTS_TOTAL = 0
for l1,l2 in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:]):
    WEIGHTS_TOTAL += l1*l2
BIASES_TOTAL = np.sum(LAYER_SIZES[1:])

# collecting the virtual batch
VBN_TOTAL = len(LAYER_SIZES[1:-1])*2
batch_size = 128
obs = env.reset()

reward_total = 0.0
VIRTUAL_BATCH = np.zeros((batch_size, STATE_SIZE))
for e in range(batch_size):
    action = np.random.randn(ACTION_SIZE)
    obs, reward, terminate, dis_reward = env.step(action)
    VIRTUAL_BATCH[e, :] = np.copy(obs)
print("VIRTUAL BATCH COLLECTED!")

def to_nn(individual):
    return individual_to_neural_network(individual=individual, layer_sizes=LAYER_SIZES, layer_activations=LAYER_ACTIVATIONS, biases=BIASES,
                                            use_vbn=VBN, virtual_batch=VIRTUAL_BATCH) 

def init_base(dimension_size):
    b = np.random.randn(1, dimension_size)
    b[0, -2] = 1.0
    b[0, -1] = 0.0
    return b

def run(population_size=20, noise_size=0.001, step_size=0.001, repetitions=2, select_percent=0.8, max_steps=1000):

    env_eo = EvolutionObjective(environment=env, repetitions=repetitions, to_model=to_nn, max_steps=max_steps) #, mirror_action=mirror_control, add_signal_data=add_signal_data)

    es_op = EsOptimiser(population_size=population_size, noise_size=noise_size, step_size=step_size, dimension_size=WEIGHTS_TOTAL+BIASES_TOTAL+VBN_TOTAL, 
                        select_percentage=select_percent, virtual_batch=VIRTUAL_BATCH, init_base_function=init_base)

    es_op.run_steps(objective=env_eo, num_runs=10000, print_out=True, run_name="Test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop', type=int, default=100, help='population size.')
    parser.add_argument('--noise', type=float, default=0.001, help='mutation scale.')
    parser.add_argument('--step', type=float, default=0.001, help='step size for updating')
    parser.add_argument('--rep', type=int, default=1, help='number of times to run each agent in the environment')
    parser.add_argument('--select', type=float, default=1.0, help='number of the best individuals to select for the next generation')
    parser.add_argument('--maxenvsteps', type=int, default=500, help='maximum number of steps an agent can take in the environment.')

    args = parser.parse_args()

    run(population_size=args.pop, 
            noise_size=args.noise, 
            step_size=args.step,
            repetitions=args.rep,
            select_percent=args.select,
            max_steps=args.maxenvsteps)
