import numpy as np
from neural_network import NeuralNetwork
from utils import individual_to_neural_network

def test_gene_to_nn():

    layer_sizes = [20, 100, 80, 4]
    layer_activations = ['tanh', 'tanh', 'tanh']
    total_weights = 20*100 + 100*80 + 80*4
    total_bias = 100 + 80 + 4
    total_vbn = 2 + 2
    gene = np.random.randn(1, total_weights + total_bias + total_vbn)

    virtual_batch = np.random.randn(128, 20)

    # convert the gene to nn
    nn = individual_to_neural_network(gene, layer_sizes, layer_activations, #multi_leg_signal, 
                                    biases=True, use_vbn=True, virtual_batch=virtual_batch) #, action_clip=0.3)
    print(nn.layers)
    # now check that the neural network weights and the gene weights match
    culm = 0
    for idx,(l1,l2) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        nn_w1 = np.round(nn.layers[idx*3].weight.data.numpy(), 3)
        ge_w1 = np.round(np.reshape(gene[0, culm:(culm + l1*l2)], (l2, l1)), 3)
        culm += l1*l2
        assert np.all(nn_w1 - ge_w1 < 0.01), "layer {} weights do not match, \n nn: \n{}, \n gene: \n{}, \n nn-size: {}, gene-size: {}".format(
                idx, nn_w1, ge_w1, np.shape(nn_w1), np.shape(ge_w1))

    # now check the biases match
    culm = 0
    for idx,l in enumerate(layer_sizes[1:]):
        nn_b1 = np.round(nn.layers[idx*3].bias.data.numpy(), 3)
        ge_b1 = np.round(np.reshape(gene[0, (total_weights + culm):(total_weights + culm + l)], (l,)), 3)
        culm += l
        assert np.all(nn_b1 - ge_b1 < 0.01), "layer {} biases do not match, \n nn: \n{}, \n gene: \n{}, \n nn-size: {}, gene-size: {}".format(
                idx, nn_b1, ge_b1, np.shape(nn_b1), np.shape(ge_b1))

    # now check the vbn
    culm = 0
    for idx,l in enumerate(layer_sizes[1:-1]):
        nn_b1 = np.round(np.array([nn.layers[2 + idx*3].gamma, nn.layers[2+idx*3].beta]), 3)
        ge_b1 = np.round(np.reshape(gene[0, (total_weights+total_bias + culm):(total_weights+total_bias + culm + 2)], (2,)), 3)
        culm += 2
        assert np.all(nn_b1 - ge_b1 < 0.01), "layer {} vbn do not match, \n nn: \n{}, \n gene: \n{}, \n nn-size: {}, gene-size: {}".format(
                idx, nn_b1, ge_b1, np.shape(nn_b1), np.shape(ge_b1))

if __name__ == "__main__":
    print("--RUNNING ALL TESTS--")
    test_gene_to_nn()
    print("TEST 1 SUCCESS")
