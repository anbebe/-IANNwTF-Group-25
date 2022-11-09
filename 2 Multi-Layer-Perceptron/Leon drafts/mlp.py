import numpy as np
import layer

class MLP:

    def __init__(self, n_inputs, n_units, n_outputs, input) -> None:
        self.hiddenLayer = layer.Layer(n_inputs, n_units, input)
        self.outputLayer = layer.Layer(n_units, n_outputs, input=np.empty([n_units,1]))
        self.output = np.zeros([n_outputs, 1])

    def forward_step(self):
        self.hiddenLayer.forward_step()
        self.outputLayer.set_input(self.hiddenLayer.a) # set input of outputLayer to output of hiddenLayer
        self.outputLayer.forward_step()
        self.output = self.outputLayer.a


    def backpropagation(self, alpha, error):
        self.outputLayer.backward_step(alpha, error) 
        self.hiddenLayer.backward_step(alpha, self.outputLayer.dx)


mlp = MLP(2, 10, 1, np.array([[1], [1]]))
epochs = 100
for i in range(epochs):
    mlp.forward_step()
    print("mlp output", mlp.output)
    error = np.array([1 - mlp.output[0]])
    print("error in epoch ",i,  error)
    mlp.backpropagation(0.005, error)


