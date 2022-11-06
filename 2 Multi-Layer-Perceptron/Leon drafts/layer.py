class Layer:
    
    def __init__(self, n_units: int, input_units: int):
        self.n_units = n_units # number of units in layer
        self.input_units = input_units # number of inputs from previous layer
        self.weights = None # initiate matrix shape(n inputs, n units) random values for the weights and zeros for the biases.
        self.layer_input = None # array with inputs from previous layer
        self.layer_preactivation = None
        self.layer_activation = None

    def forward_step(self):
        output = None # generate numpy array with all zeros
        for i in range(len(self.n_units)):
            temp = self.layer_input@self.weights[i] # matrix multiplication with input and weights of i-th unit
            if temp > 0:
                output[i] = 1

    def backward_step(self):
        pass





