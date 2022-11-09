# still everything to import
import numpy as np

class Layer:
    
    def __init__(self,  n_inputs: int, n_units: int, input):
        self.n_units = n_units # number of units in layer
        self.n_inputs = n_inputs # number of inputs from previous layer

        self.b = np.random.rand(n_units, 1) # biases - m:1
        self.w = np.random.rand(n_inputs, n_units) # weights - n:m

        self.x = input # input n:1
        self.z = np.zeros([n_units, 1]) # array with all inputs for ReLu function m:1
        self.a = np.zeros([n_units, 1]) # array with all outputs for ReLu function m:1
        
        self.dx = np.zeros([n_inputs, 1]) # n:1

    def set_input(self, input):
        self.x = input

    def ReLU(self, z):
        return np.maximum(0,z)

    def dReLU(self, z):
        return z > 0

    def forward_step(self):
        #print("shape w", self.w.T.shape)
        self.z = self.w.T.dot(self.x) + self.b # activation input -> m:1 = m:n o n:1
        #print("shape z", self.z.shape)
        self.a = self.ReLU(self.z) # activation output -> um:1 = ReLU(m:1)
        #print("shape output", self.a.shape)
        
    def backward_step(self, alpha, error):
        dz = error * self.dReLU(self.z) # gradient activation input -> m:1 = m:1 * m:1
        # print("shape dz", dz.shape)
        
        dw = self.x.dot(dz.T)# gradient weights -> n:m = n:1 o 1:m
        # print("gradients weights", dw)
        # print("shape dw", dw.shape)
        
        db = dz # gradient biases -> m:1 = m:1 * 1
        
        self.dx = self.w.dot(dz) # gradients input -> n:1 = n:m o m:1
        # print("shape dx", self.dx.shape)

        self.w = self.w - alpha*-dw
        self.b = self.b - alpha*-db





