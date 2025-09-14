import numpy as np
import math
import random

class MLP:
    def __init__(self, num_inputs: int = 2, num_hidden: list = [3, 5], num_outputs = 2, s: int = 123):
        # get initial parameters
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        # create an internal representation of layers, each index stores the number of neurons in that layer 
        layers = [self.num_inputs] + num_hidden + [self.num_outputs] 
        np.random.seed(s) 
        # set weights
        weights = [] 
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        # initialize activation value matrix
        activations = [] 
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        #initialize derivatives
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def _mse(self, output, target):
        return np.average((output-target) ** 2)

    
    def forward_propagate(self, inputs: list):
        if len(inputs) > self.num_inputs:
            print("Check length of input vector.")
            return None
        # set activations        
        activations = inputs
        self.activations[0] = inputs
        #vectorize activation function
        sigmoid = np.vectorize(self._sigmoid)
        # compute weighted sum matrix
        for i,w in enumerate(self.weights):
            weighted_sum = np.dot(activations, w) # take the dot product of the output of previous layer with the weights
            activations = sigmoid(weighted_sum) # set activations by applying sigmoid
            self.activations[i+1] = activations # save the activation values
        
        # after the for loop the activations represent the activation states of the output layer
        return activations

    def back_propagation(self, error, verbose=False):
        sigmoid = np.vectorize(self._sigmoid)
        # dE/dW_i = (y - a_[i+1]) s' (h_[i+1])) a_i
        #s'(h_[i+1]) = s(h_[i+1]) (1 - s(h_[i+1]))
        #s(h_[i+1]) = a_[i+1]
        #dE/dW_[i-1] = (y - a_[i+1]) s'(h_[i+1])) W_i s'(h_i) a_[i-1]
        for i in reversed(range(len(self.derivatives))):
            activation = self.activations[i+1]
            delta = error * (activation) * (1 - activation)
            delta = delta.reshape(delta.shape[0], -1).T # reshape into a row matrix
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0], -1) # reshape into a column matrix
            self.derivatives[i] = np.dot(current_activations, delta)
            #update error
            error = np.dot(error, self.weights[i].T)

            if verbose:
                print(f"Derivative at W{i}: {self.derivatives[i]}")
        
        return error # returns the error backpropagated till the input layer

    def gradient_descent(self, learning_rate: int):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            # print(f"Original Weights: W_{i} => {weights}")
            derivatives = self.derivatives[i]
            weights = weights + derivatives * learning_rate
            # print(f"Updated Weights: W_{i} => {weights}")
            self.weights[i] = weights
    
    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_error = 0
            for inp, target in zip(inputs, targets):
                output = self.forward_propagate(inp)
                error = target - output

                self.back_propagation(error=error)

                self.gradient_descent(learning_rate=learning_rate)

                sum_error += self._mse(target, output)

            print(f"Error in Epoch {i+1}: {sum_error/len(inputs)}")

        

if __name__ == "__main__":
    # create MLP
    mlp = MLP(num_inputs=2, num_hidden=[5], num_outputs=1, s=29)
    # take inputs
    inputs = np.array([[random.random()/2 for i in range(2)] for i in range(1000)]) # 1000 training samples
    targets = np.array([[i[0] + i[1]] for i in inputs])

    mlp.train(inputs, targets, epochs=50, learning_rate=2)
    
    #test

    inputs = np.array([0.1, 0.2])
    print(mlp.forward_propagate(inputs))
    