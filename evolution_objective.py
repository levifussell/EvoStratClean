import numpy as np

class EvolutionObjective():
    """
    This class converts an rl objective function for RL into an objective function for
    evolution algorithms. It adds a parameter for a certain number of repetitions and
    also incorporates the weights changing from a vector into a model (such as a feed-
    forward model).
    """


    def __init__(self, environment, repetitions, to_model,
            weight_decay=0.0, max_steps=1000):
        self.environment = environment
        self.repetitions = repetitions
        self.to_model = to_model
        self.weight_decay = weight_decay
        self.max_steps = max_steps

    def compute(self, individual, no_terminate=False, add_to_buffer=False): #, swap_signal=False):
        
        # first convert the individual into a model
        model = self.to_model(individual)#, swap_signal=swap_signal)

        # weight decay term
        weight_sum = 0.0
        for l in model.layers:
            if l.__class__.__name__.find('Linear') != -1:
                weight_sum += np.sum(np.abs(l.weight.data.numpy()))

        # run the individual through the object a certain number of times
        fitnesses = np.zeros(self.repetitions)
        for r in range(self.repetitions):
            fitnesses[r] = self.get_environment_reward(model, no_terminate=no_terminate)

        fit_mean = np.mean(fitnesses) - self.weight_decay * weight_sum
        return fit_mean
    
    def get_environment_reward(self, model, no_terminate=False): #, add_to_buffer=False):

        obs = self.environment.reset()
        reward_total = 0.0

        for e in range(self.max_steps):

            action = model.forward(obs)

            obs, reward, terminate, dis_reward = self.environment.step(action)

            reward_total += reward

            if terminate and not no_terminate:
                break

        return reward_total
