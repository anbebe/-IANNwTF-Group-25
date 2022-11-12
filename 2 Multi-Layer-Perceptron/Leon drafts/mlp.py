import numpy as np
import layer
import data

class MLP:

    def __init__(self, n_layers, units_per_layer) -> None:
        self.n_layers = n_layers
        self.units_per_layer = units_per_layer
        
        self.mlp = [] # create mlp with n_layers without input layer
        
        for i in range(n_layers):
            self.mlp.append(layer.Layer(units_per_layer[i], units_per_layer[i+1], np.empty([units_per_layer[i],1]))) # create layers of mlp

        self.mlp = np.array(self.mlp)

        self.output = np.zeros([units_per_layer[-1],1]) # define output 
        print("output shape constructor", self.output.shape)

        #self.hiddenLayer = layer.Layer(n_inputs, n_units, input)
        #self.outputLayer = layer.Layer(n_units, n_outputs, input=np.empty([n_units,1]))
        #self.output = np.zeros([n_outputs, 1])

    def print(self):
        for i in range(self.n_layers):
            print(self.mlp[i])

    def forward_step(self, input):
        x = input
        for i in range(self.n_layers):
            self.mlp[i].set_input(x)
            self.mlp[i].forward_step()
            x = self.mlp[i].a

        self.output = x
        print("output in function", x.shape)
        print("output in function",self.output.shape)

       # self.hiddenLayer.forward_step()
        #self.outputLayer.set_input(self.hiddenLayer.a) # set input of outputLayer to output of hiddenLayer
        #self.outputLayer.forward_step()
        #self.output = self.outputLayer.a


    def backpropagation(self, alpha, target):

        dx = error
        
        for i in range(self.n_layers):
            self.mlp[self.n_layers-1-i].backward_step(alpha, dx) # backward step of current layer i starting at last layer
            dx = self.mlp[self.n_layers-1-i].dx # gradient of inputs of current layer starting at last layer

        #self.outputLayer.backward_step(alpha, error) 
        #self.hiddenLayer.backward_step(alpha, self.outputLayer.dx)


inputs, targets = data.generate_data()
mlp = MLP(2, np.array([1, 10, 1]), np.array([[1], [1]]))
inputs, targets = data.generate_data()
epochs = 5
for i in range(epochs):
    mlp.forward_step(inputs[i])
    print("mlp output", mlp.output)
    error = np.array([1 - mlp.output[0]])
    print("error in epoch ",i,  error)
    mlp.backpropagation(0.05, error)


