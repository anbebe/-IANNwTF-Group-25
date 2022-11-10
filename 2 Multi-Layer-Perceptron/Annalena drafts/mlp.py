import numpy as np
import matplotlib.pyplot as plt

### data functions ###

def generate_data():
    x = np.random.rand(100)
    t = np.array([y**3 - y**2 for y in x])
    return x, t

def plot_data(x, t):
    plt.plot(x,t,'bo')
    x_2 = np.linspace(0,1,100)
    y_2 = x_2**3 - x_2**2
    plt.plot(x_2,y_2,'r-')
    plt.show()

class Layer:

    def __init__(self, n_units, input_units):
        assert type(n_units) == int and type(input_units) == int
        self.n_units = n_units
        self.i_units = input_units
        self.bias = np.zeros(n_units)
        self.weights = np.random.rand(self.i_units, self.n_units)
        self.layer_input = []
        self.layer_preact = []
        self.layer_act = []
        self.next_layer = None
        self.lr = 0.01
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
        self.layer_input = x
        self.layer_preact = np.matmul(self.layer_input, self.weights) + self.bias
        self.layer_act = np.maximum(self.layer_preact, np.zeros(self.layer_preact.shape))
        return self.layer_act

    def backward_step(self, targets):
        # compute gradients and reuse ame calculations
        tmp = self.relu_d(self.layer_preact)
        act = self.activation_d(targets)
        d_bias = self.relu_d(self.layer_preact) * self.activation_d(targets)
        d_weights = np.transpose(self.layer_input) * d_bias
        self.d_inputs = d_bias * np.transpose(self.weights)

        # update parameters
        self.weights = self.weights - self.lr*d_weights
        self.bias = self.bias - self.lr*d_bias




class MLP:

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
        out = x
        for i in range(len(self.network)):
            out = self.network[i].forward_step(out)
        return out
    
    def backpropagation(self, predict , targets):
        l = np.mean(0.5 * (predict - targets)**2)
        self.loss.append(l)
        for i in range((len(self.network)-1), -1, -1):
            self.network[i].backward_step(targets)

    def print_network(self):
        for i in range(len(self.network)):
            print("Layer ", i, ", input units: ", self.network[i].i_units, " number units: ", self.network[i].n_units)



if __name__ == "__main__":

    inputs, targets = generate_data()
    # plot_data(inputs, targets)

    mlp = MLP(n_layer=1, input_units=1, output_units=1, hidden_units=10)
    mlp.print_network()

    loss = []

    epochs = 1000

    for e in range(epochs):
        for i in range(len(inputs)):
            out = mlp.forward_step(np.expand_dims(np.array([inputs[i]]), axis=0))
            mlp.backpropagation(out, targets[i])
        loss.append(np.mean(mlp.loss))
        mlp.loss = []
        if e % 100 == 0:
            print(e)

    
    plt.plot(range(epochs),loss,'ro')
    plt.show()



