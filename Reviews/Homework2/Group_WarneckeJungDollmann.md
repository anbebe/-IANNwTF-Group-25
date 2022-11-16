# Homework Review Group_WarneckeJungDollmann


## Positive Remarks

1. good usage of numpy functions
```python
preactivation_dertivate = np.heaviside(self.layer_preactivation, 0)
```
2. simple way to initialize structure of network with one list

3. good generalized implementation of MLP with list structure of layers

4. generally very nice good which works 

***
## Possible improvements

1. The backward step could be optimized through only one calcaluting the follwoing product and reusing it in the computation of the gradients

```python
np.multiply(preactivation_derivative, next_layer_derivative)
```

2. The vectors x and t could be defined or referenced once more during training as one has to search for the variable in the code above

3. The learning rate could passed through an argument


***
## Not correct
1. Incorrect feeding of input to the network. For 1000 epochs the same data is fed over and over again. The task was to have one input, instead you have 100. 

```python
for i in range(epochs):
    # getting the output for every input
    output = mlp.forward_step(x)
    current_loss = MSE(output,t)
    loss.append(current_loss)
    # backpropagation
    mlp.backpropagation(MSE_derivative(output,t))
```