import numpy as np

class Neuron:
    def __init__(self, weights, activation, bias = 0):
        self.weights = weights
        self.activation = activation
        self.bias = bias
        
    def output(self, x):
        dp = np.dot(self.weights, x)
        return self.activation(dp + self.bias)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = [1.2, 4.5, 6.7, 8]
w = [3, 2, 1, 4]

neuron = Neuron(w, sigmoid)
result = neuron.output(x)
print(result)







