# Review Group 33

***
## Strucural differences 
1. Class for OutputLayer and HiddenLayer seperately

***
## Good points
1. nice reference to mathematical theory e.g. with weights_jacobi_matrix

***
## Not correct
1. biases of HiddenLayer have wrong dimension for forward step in line 60

```python
self.bias = np.array([0]*n_units)
```

**Try instead**
```python
self.bias = np.array(1,n_units)
```

2. no usage of line 68
```python
self.activations_last = X
```

3. backward step uses derivative of sigmoid fundtion even though relu function was used in forward step in line 79/80
```python
self.errors = np.dot(next_layer_weights.T, next_layer_errors) * sigmoid_prime(np.dot(self.weights, last_layer_activations.T))
```



