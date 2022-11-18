# Review for Group Lederer

## Good points
1. it works
2. mathematical formulas nicely documented


## Points to improve
1. almost no comments -> not very nice to read the code
2. very cryptic variable names e.g. d_L_d_a_layer ... -> not self explanatory
3. does not work for more than 1 hidden layer
4. learning rate can't be modified after intantiating the object MLP
5. possible to store compututation of codeline below to save computational cost
```python
np.where(self.n_preactivation > 0,1,0)
```
