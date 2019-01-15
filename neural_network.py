import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

class VbnLayer(nn.Module):

    def __init__(self, input_size, gamma, beta):
        super(VbnLayer, self).__init__()
        
        self.gamma = gamma
        self.beta = beta

    def forward(self, x, virtual_batch):
        vb_mean = torch.mean(virtual_batch, 0)
        vb_var = torch.var(virtual_batch, 0)
        
        # normalise
        x_norm = (x - vb_mean)/torch.sqrt(vb_var + 1e-8)
        
        # undo normalisation
        x_norm *= self.gamma
        x_norm += self.beta
        return x_norm

class NeuralNetwork(nn.Module):

    def __init__(self, layer_sizes, layer_activations, weights, biases,
                virtual_batch=None, vbn_params=None):
        super(NeuralNetwork, self).__init__()

        self.virtual_batch = virtual_batch
        self.vb_mean = np.mean(virtual_batch, 0)
        self.vb_std = np.sqrt(np.var(virtual_batch, 0) + 1e-8)

        self.layers = []
        for idx,(l1,l2) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            l = nn.Linear(l1, l2)

            # set weights
            l.weight.data.copy_(torch.reshape(torch.FloatTensor(weights[idx]), (l.weight.data.shape[0], l.weight.data.shape[1])))

            # set biases
            l.bias.data.copy_(torch.FloatTensor(biases[idx]))

            self.layers.append(l)

            if layer_activations[idx] == 'relu':
                self.layers.append(nn.ReLU(True))
            elif layer_activations[idx] == 'tanh':
                self.layers.append(nn.Tanh())
            elif layer_activations[idx] == 'sigmoid':
                self.layers.append(nn.Sigmoid())

            if self.virtual_batch is not None and idx < len(layer_activations)-1:
                self.layers.append(VbnLayer(input_size=l2, 
                                            gamma=vbn_params[idx, 0],
                                            beta=vbn_params[idx, 1]))
    def forward(self, x):

        x_t = Variable(torch.from_numpy(np.float32(x)[None,:]))

        if self.virtual_batch is not None:
            v_t = Variable(torch.from_numpy(np.float32(self.virtual_batch)))

        for l in self.layers:
            # do virtual batchnorm if not None
            if self.virtual_batch is not None:
                if l.__class__.__name__.find('Vbn') != -1:
                    x_t = l(x_t, v_t)
                else:
                    x_t = l(x_t)
                    v_t = l(v_t)
            else:
                x_t = l(x_t)

        action = x_t.detach().data.numpy()[0]
        
        return action.tolist()
