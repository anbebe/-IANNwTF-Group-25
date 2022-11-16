import numpy as np
import matplotlib.pyplot as plt

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



x, t = generate_data()
plot_data(x,t)