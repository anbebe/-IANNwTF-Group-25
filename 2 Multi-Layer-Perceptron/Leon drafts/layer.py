# still everything to import
import numpy as np

class Layer:
    
    def __init__(self,  n_inputs: int, n_units: int, input):
        self.m = n_units # number of units in layer: m
        self.n = n_inputs # number of inputs from previous layer: n

        self.b = np.random.rand(self.m, 1) # biases -> m:1
        self.w = np.random.rand(self.n, self.m) # weights -> n:m

        self.x = input # input -> n:1
        self.z = np.zeros([self.m, 1]) # activation input -> m:1
        self.a = np.zeros([self.m, 1]) # activation output -> m:1
        
        self.dx = np.zeros([self.n, 1]) # gradient input -> n:1

    def set_input(self, input):
        self.x = input

    def ReLU(self, z):
        return np.maximum(0,z)

    def dReLU(self, z):
        return z > 0

    def forward_step(self):
        self.z = self.w.T.dot(self.x) + self.b # activation input -> m:1 = m:n o n:1
        self.a = self.ReLU(self.z) # activation output -> m:1 = ReLU(m:1)
        
    def backward_step(self, alpha, error):
                
        dz = error * self.dReLU(self.z) # gradient activation input -> m:1 = m:1 * m:1
        dw = self.x.dot(dz.T)# gradient weights -> n:m = n:1 o 1:m        
        print("dw", dw)
        db = dz # gradient biases -> m:1 = m:1 * 1
        self.dx = self.w.dot(dz) # gradients input -> n:1 = n:m o m:1

        self.w = self.w - alpha*dw
        self.b = self.b - alpha*db





