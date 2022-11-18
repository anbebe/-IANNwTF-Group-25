# Review for Group Lederer

## Good points
1. it works
2. mathematical formulas nicely documented


## Points to improve
1. almost no comments -> not very nice to read the code
3. very cryptic variable names e.g. d_L_d_a_layer ... -> not self explanatory
4. does not work for more than 1 hidden layer
5. pass number of inputs and number of hidden layers as parameters in MLP
6. learning rate can't be modified after intantiating the object MLP
7. would be nice and more intuitive to label plots
8. possible to store compututation of codeline below to save computational cost
```python
np.where(self.n_preactivation > 0,1,0)
```
