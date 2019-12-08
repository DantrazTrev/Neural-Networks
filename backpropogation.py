def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def cost_derivative(self, output_activations, y):
        return (output_activations-y)

#The backpropogation function 
def backprop(net, x, y):
        nabla_b = [np.zeros(b.shape) for b in net.biases] #intialising biases 
        nabla_w = [np.zeros(w.shape) for w in net.weights]  #intialising weights
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(net.biases, net.weights):
                z = np.dot(w, activation)+b`   
            zs.append(z)
            activation = sigmoid(z)          #calculating activations for the given layer
            activations.append(activation)
        delta = net.cost_derivative(activations[-1], y) *(sigmoid_prime(zs[-1]))      #calculating delta for the output layer
        nabla_b[-1] = delta                                                           #calculating delta[b] for output layer
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())                      #calculating delta[w] for output layers
        
        
        """
        Similarly calculating
        delta[b],delta[w] for other hidden layers
        
        """
        for l in range(2, net.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(net.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (net,nabla_b, nabla_w)                                             #returning network, delta[b],delta[w] for the whole network
