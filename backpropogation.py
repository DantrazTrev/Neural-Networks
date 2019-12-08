import numpy as np

def sigmoid(z):
return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
return sigmoid(z)*(1-sigmoid(z))
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
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(net.biases, net.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = net.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, net.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(net.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (net,nabla_b, nabla_w)
