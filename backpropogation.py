import numpy as np

def sigmoid(z):                      #  This is funtion which gives output between 0 and 1 
                                     #  it represents the activation of that neuron.
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))    # This gives the derivative of the sigmoid function

#The backpropogation function 
def backprop(net, x, y):
        '''
        This function performs Back Propogation of a Neural Network. 
        It takes neural net as OBJECT and training data as List as argument 
        and returns neural net as OBJECT and derivative of cost wrt to Bias (nabla_b) and weights(nabla_w) as numpy array.
        Use:
        `net,nabla_b, nabla_w = backprop(net, x, y)`
        '''
        nabla_b = [np.zeros(b.shape) for b in net.biases]
        nabla_w = [np.zeros(w.shape) for w in net.weights]
        #feedforward
        activation = x
        activations = [x]  #list to store all the activations, layer by layer
        zs = []            #list to store all z vectors, layer by layer
        for b, w in zip(net.biases, net.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        # transpose() returns the numpy array with the rows as columns and columns as rows
        delta = net.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in range(2, net.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(net.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (net,nabla_b, nabla_w)                                       #Return ''(nabla_b, nabla_w)'' representing the
                                                                            #gradient for the cost function.
