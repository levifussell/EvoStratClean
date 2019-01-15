import numpy as np
from neural_network import NeuralNetwork

class OptSgd():
    
    def __init__(self, step_size):
        self.step_size = step_size

    def step(self, gradient):
        return self.step_size * gradient

class OptAdam():

    def __init__(self, step_size, wind_mom_rate, wind_grad_rate):
        self.step_size = step_size
        self.wind_mom_rate = wind_mom_rate
        self.wind_grad_rate = wind_grad_rate
        self.wind_mom = None
        self.wind_grad = None

    def step(self, gradient):

        if self.wind_mom is None:
            self.wind_mom = np.zeros_like(gradient)
        if self.wind_grad is None:
            self.wind_grad = np.zeros_like(gradient)

        self.wind_mom = self.wind_mom_rate*self.wind_mom + (1.0 - self.wind_mom_rate)*gradient
        self.wind_grad = self.wind_grad_rate*self.wind_grad + (1.0 - self.wind_grad_rate)*(gradient**2)
        return self.step_size*(self.wind_mom/(np.sqrt(self.wind_grad)+1e-8))

def individual_to_neural_network(individual, layer_sizes, layer_activations,
                                biases=True, use_vbn=True, virtual_batch=None):
    '''
    Converts a 'chromosome'/individual to a PyTorch neural network module.
    '''

    # if biases is true, we first take off all the biases from the end
    bias_section = np.sum(layer_sizes[1:]) # sum up all nodes except the last
    vbn_section = len(layer_sizes[1:-1])*2
    total_section = bias_section+vbn_section
    if biases:
        if use_vbn:
            weights = individual[:, :-total_section]
            biases = individual[0, -total_section:(-total_section+bias_section)]
            vbn = individual[:, (-total_section+bias_section):]
            assert len(vbn[0,:]) + len(weights[0,:]) + len(biases) == len(individual[0, :]), "length mismatch"
        else:
            weights = individual[:, :-total_nodes]
            biases = individual[0, -total_nodes:]
            vbn = None
            assert len(weights) + len(biases) == len(individual[0, :]), "length mismatch"
    else:
        if use_vbn:
            weights = individual[:, :-vbn_section]
            vbn = individual[:, -vbn_section:]
            assert len(weights) + len(vbn) == len(individual[0, :]), "length mismatch"
        else:
            weights = individual[:]
            vbn = None
        biases = np.zeros(total_nodes) + 0.01 # we fill the biases with 0.01 because maybe that's more interesting

    culm_w = 0
    culm_b = 0
    layer_weights = []
    layer_biases = []
    for l1,l2 in zip(layer_sizes[:-1], layer_sizes[1:]):

        weight_count = l1*l2
        w = np.array(weights[0, culm_w:(culm_w+weight_count)])
        w = np.reshape(w, (l1,l2))
        culm_w += weight_count
        layer_weights.append(w)
        
        b = biases[culm_b:(culm_b+l2)]
        culm_b += l2
        layer_biases.append(b)

    if use_vbn:
        vbn = np.reshape(vbn, (-1, 2))

    torch_nn = NeuralNetwork(layer_sizes, layer_activations, layer_weights, layer_biases,
                            virtual_batch=virtual_batch, vbn_params=vbn)
    return torch_nn
