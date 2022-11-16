import numpy as np
import matplotlib.pyplot as plt
from data import Data
from layer import Layer


class MLP:
    '''
    Mulitlayer Perceptron Class, initialising an MLP with at least one hidden layer and assuming MSE loss
    '''

    def __init__(self, n_layer, input_units, output_units, hidden_units):
        ''' assert MLP with at least one input and one output layer and n_layer number of hidden layer'''

        self.network = []
        # input layer
        self.network.append(Layer(n_units=hidden_units, input_units=input_units))
        # add given number (n_layer) of hidden layer to the network and connect each one to the next
        for i in range(n_layer-1):
            l = Layer(n_units=hidden_units, input_units=self.network[i].n_units)
            self.network[i].set_next_layer(l)
            self.network.append(l)
        # output layer
        output_layer = Layer(n_units=output_units, input_units=self.network[-1].n_units)
        self.network[-1].set_next_layer(output_layer)
        self.network.append(output_layer)
        self.loss = []

    def forward_step(self,x):
        '''
        Forward step of the network, returning the output of the last layer.
        '''
        out = x
        for i in range(len(self.network)):
            out = self.network[i].forward_step(out)
        return out
    
    def backpropagation(self, predict , targets):
        '''
        Doing the backward step from layer to layer and saving the MSE loss per input
        '''
        l = np.mean(0.5 * (predict - targets)**2)
        self.loss.append(l)
        for i in range((len(self.network)-1), -1, -1):
            self.network[i].backward_step(targets)

    def print_network(self):
        '''
        Visualise network with each layer and its units
        '''
        for i in range(len(self.network)):
            print("Layer ", i, ", input units: ", self.network[i].i_units, " number units: ", self.network[i].n_units)



if __name__ == "__main__":
    # create data
    inputs, targets = Data.generate_data()
    Data.plot_data(inputs, targets)
    
    # create MLP
    mlp = MLP(n_layer=1, input_units=1, output_units=1, hidden_units=10)
    mlp.print_network()

    loss = []

    epochs = 1000

    # Training of the network
    for e in range(epochs):
        for i in range(len(inputs)):
            out = mlp.forward_step(np.expand_dims(np.array([inputs[i]]), axis=0))
            mlp.backpropagation(out, targets[i])
        loss.append(np.mean(mlp.loss))
        mlp.loss = []
        if e % 100 == 0:
            print(e)

    # visualise training process
    plt.plot(range(epochs),loss,'ro')
    plt.show()



