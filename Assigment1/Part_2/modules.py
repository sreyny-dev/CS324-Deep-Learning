import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer. 
        TODO: Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        self.x = None
        weight_std = np.sqrt(2.0 / in_features)
        self.params = {
            'weight': np.random.normal(0.0, weight_std, (in_features, out_features)),
            'bias': np.zeros((1, out_features))
        }
        self.grads = {
            'weight': np.zeros((in_features, out_features)),
            'bias': np.zeros((1, out_features))
        }

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        TODO: Implement the forward pass.
        """
        self.x = x
        forward_linear = np.dot(x, self.params['weight']) + self.params['bias']
        return forward_linear

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """
        self.grads['weight'] = np.dot(self.x.T, dout)  # dL/dW = x . dL / dout_layer
        self.grads['bias'] = np.sum(dout, axis=0, keepdims=True)  # dL/db = dL/dout_layer
        dout = np.dot(dout, self.params['weight'].T)
        return dout


class ReLU(object):
    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        TODO: Implement the forward pass.
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        TODO: Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        return dout * (self.x > 0).astype(float)


class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        TODO: Implement the forward pass using the Max Trick for numerical stability.
        """
        # x.shape -> (N,C)
        exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        # softmax equation
        softmax_output = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        return softmax_output

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        TODO: Keep this in mind when implementing CrossEntropy's backward method.
        """
        return dout


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        TODO: Implement the forward pass.
        """
        n_samples = y.shape[0]
        x_clipped = np.clip(x, 1e-10, 1.0)
        likelihood_logprobs = -np.log(x_clipped[range(n_samples), y.argmax(axis=1)])
        return np.sum(likelihood_logprobs) / n_samples  # loss in scalar

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        TODO: Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        grad = x - y
        return grad
