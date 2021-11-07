import numpy as np
import math

class Forward:

    def __init__(self, data):
        self.data = np.array(data)

    def sigma(self, weight, bias):
        return np.sum(self.data * weight) + bias

    def sigmoid(self, output):
        return 1 / (1 + np.exp(-output))

    def softmax(self, y):
        output = []
        for i in y:
            output.append(np.exp(i) / np.sum(np.exp(y)))
        return output

    def error(self, target, output):
        return -(target * np.log10(output))







