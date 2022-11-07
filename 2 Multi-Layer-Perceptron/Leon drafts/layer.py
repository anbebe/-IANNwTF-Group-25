# still everything to import
import numpy as np

class Layer:
    
    def __init__(self, n_units: int, input_units: int, layer_input, layer_activation):
        self.n_units = n_units # number of units in layer
        self.input_units = input_units # number of inputs from previous layer
        self.weights = None # initiate matrix shape(n inputs, n units) random values for the weights and zeros for the biases.
        self.layer_input = None # array with outputs from previous layer
        self.layer_preactivation = None # array with all inputs for ReLu function
        self.layer_activation = layer_activation # array with all activations of next layer

    def ReLU(z):
        return np.maximum(0,z)

    def forward_step(self):
        output = None # generate numpy array with all zeros
        z = self.weights.dot()

        return output

    def backward_step(self):
        # how much the weights influence the outcome
        gradients_weights = None # self.layer_input * derivative of relu(self.activations) * derivitave of activation of next layer
        
        # how much the bias influences the outcome
        gradients_bias = None # derivative of relu(self.activations) * derivitave of activation of next layer
        
        # how much the input influences the outcome
        gradients_input = None # for k in len(self.layer_input) sum(self.weights[k]* derivative of relu(self.activations)*derivative of activation of next layer

        self.weights = None # update the parameters in right direction of the gradients





