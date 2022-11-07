# still everything to import
import numpy as np

class Layer:
    
    def __init__(self, n_units: int, n_input_units: int, layer_input, layer_activation):
        self.n_units = n_units # number of units in layer
        self.n_input_units = n_input_units # number of inputs from previous layer
        self.b = np.random.rand(n_units)
        self.w = np.random.rand(n_input_units, n_units) # initiate matrix shape(n inputs, n units) random values for the weights and zeros for the biases.
        self.x = layer_input # array with outputs from previous layer
        self.z = np.zeros(n_input_units) # array with all inputs for ReLu function
        self.a = np.zeros(n_units) # array with all activations

    def ReLU(self, z):
        return np.maximum(0,z)

    def dReLU(self, z):
        return z > 0

    def forward_step(self):
        self.z = self.w.dot(self.x) + self.b
        self.a = self.ReLU(self.z)
        
    def backward_step(self, alpha):
        # how much the weights influence the outcome
        dz = self.dReLU(self.z) # multiplied by partial derivate to activation
        dw = dz.dot(self.x.T) # self.layer_input * derivative of relu(self.activations) * derivitave of activation of next layer
        db = dz

        self.w = self.w - alpha*dw
        self.b = self.b - alpha*db





