import numpy as np

class Layer:

    def __init__(self, n_units, input_units):
        '''
        Initialise all necssary parameters for one layer
        '''
        assert type(n_units) == int and type(input_units) == int
        self.n_units = n_units
        self.i_units = input_units
        self.bias = np.zeros(n_units)
        self.weights = np.random.rand(self.i_units, self.n_units)
        self.layer_input = []
        self.layer_preact = []
        self.layer_act = []
        # each layer know the next one, for the backward step, if none its the output
        self.next_layer = None
        self.lr = 0.01
        # saves the partial derivative of the Loss w.r.t. the layers activation for the next backpropagation step
        self.d_inputs = 0

    def relu_d(self, x):
        '''
        calculates derivative of relu
        '''
        return np.where(x>0, 1, 0)
        

    def activation_d(self, targets):
        '''
        calculates derivate of Loss (we know its MSE) w.r.t to activation and therefore from post layer
        '''
        if self.next_layer == None:
            return (self.layer_act - targets)
        else:
            return self.next_layer.d_inputs

    def set_next_layer(self, l):
        self.next_layer = l


    def forward_step(self, x):
        '''
        Forward step for one layer, saving the preactivation step (sum of input and weights) and the activation
        of ot with ReLu, returns activation
        '''
        self.layer_input = x
        self.layer_preact = np.matmul(self.layer_input, self.weights) + self.bias
        self.layer_act = np.maximum(self.layer_preact, np.zeros(self.layer_preact.shape))
        return self.layer_act

    def backward_step(self, targets):
        '''
        Calculates gradients for weights, bias and layer based on Relu and update weights and bias
        depending on the learning rate
        '''
        tmp = self.relu_d(self.layer_preact)
        act = self.activation_d(targets)
        d_bias = self.relu_d(self.layer_preact) * self.activation_d(targets)
        d_weights = np.transpose(self.layer_input) * d_bias
        self.d_inputs = d_bias * np.transpose(self.weights)

        # update parameters
        self.weights = self.weights - self.lr*d_weights
        self.bias = self.bias - self.lr*d_bias