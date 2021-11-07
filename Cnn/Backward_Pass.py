import numpy as np

class Backward:
    ln = None

    def __init__(self, ln):
        self.ln = ln


    def deltaWeightSigmoid(self, error, Fy, Fo):
        print(Fy)
        print(Fo)
        return self.ln * np.array([(error * Fy * (1 - Fy))]) * Fo

    def deltaBiasSigmoid(self, error, Fy):
        return self.ln * (error * Fy * (1 - Fy))

    def deltaWeightSoftmax(self, error, Fy, Fo, Si):
        output = np.array([])
        # output = np.array([0] * len(Fy))
        for j in range(len(Fy)):
            for o in Fo:
                if j == Si:
                    output = np.append(output, [self.ln * np.array([(error[j] * Fy[Si] * (1 - Fy[j]))]) * Fo])
                else:
                    output = np.append(output, [self.ln * np.array([(error[j] * (-Fy[Si] * Fy[j]))]) * Fo])
        return output

    def deltaBiasSoftmax(self, error, Fy, Si):
        output = np.array([])
        for j in range(len(Fy)):
            if j == Si:
                output = np.append(output, self.ln * (error[j] * Fy[Si] * (1 - Fy[j])))
            else:
                output = np.append(output, self.ln * (error[j] * (-Fy[Si] * Fy[j])))
        return output

    def weightNew(self, delta, w):
        return w + delta

    def biasNew(self, delta, b):
        return b + delta

    def errorHidden(self, w, b, error):
        return np.sum(np.dot(error, w)) + b