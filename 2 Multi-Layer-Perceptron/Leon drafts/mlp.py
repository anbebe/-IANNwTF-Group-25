# still to import

import layer

class MLP:

    def __init__(self, n_layers: int, n_units_per_layer) -> None:
        self.n_layers = n_layers
        self.layers = [layer.Layer(n_units_per_layer[n]) for n in n_layers] # create mlp with n_layers and n_units_per_layer 
        
    def forward_step(self):
        # combining forward step of 
        pass

    def backpropagation(self):
        pass