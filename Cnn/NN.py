import numpy
import numpy as np
import Forward_Pass as Fp
import Backward_Pass as Bp
import random


class MLP:
    valid = True
    w = []
    b = []
    o = []
    softmax = []
    error = []
    eh = []
    bp = Bp.Backward(0.1)
    hiddenOutput = []

    def __init__(self, data, label, hidden, perc):
        if hidden == len(perc):
            self.hidden = hidden
            self.perc = perc
            self.Y_output = [0] * len(np.unique(label))
            self.label = label
            self.data = data
            self.hiddenOutput.insert(0, data)
            self.hiddenOutput += [[0] * perc[i] for i in range(self.hidden)]
        else:
            self.valid = False

    def setWeight(self):
        print("Hidden Output", self.hiddenOutput)
        # jumlahInput = len(self.data)
        # for i in range(self.hidden):
        #     for p in range(self.perc[i]):
        #         # self.w += [[round(random.random(), 2)] * jumlahInput]
        #           self.b += [0]
        #     jumlahInput = self.perc[i]

        self.w += [0.07, 0.07], [0.27, 0.27]
        # self.w += [0.07, 0.07], [1.79384931, 1.79384931]

        self.eh = [[0] * len(self.w)]

        # for i in range(len(self.Y-output)):
        #     # self.w += [[round(random.random(), 2)] * jumlahInput]
        #     self.w += [0.27, 0.27]

        self.w += [0.27, 0.30], [0.30, 0.27]
        # self.w += [0.27, 0.30], [0.8478555826328623, 1.1700549995394731]

        for p in self.perc:
            self.o += [[0] * p]

        for p in self.perc:
            self.b += [[0] * p]
        self.b += [[0] * len(self.Y_output)]

        # self.b += [[0.0, 1.5238493147182282], [0.0, 1.0241383452039505]]
        print("Data", self.data)
        print("hidden Layer", self.hidden)
        print("Neuron", self.perc)
        print("Eh", self.eh)
        print("Output", self.Y_output)
        print("Weight", self.w)
        print("Bias", self.b)
        print("Jumlah Perc", self.o)

    def forwardProses(self):
        data = self.data
        fp = Fp.Forward(data)
        for i in range(self.hidden):
            for f in range(self.perc[i]):
                self.o[i][f] = fp.sigmoid(fp.sigma(self.w[f], self.b[i][f]))
            data = self.o[i]
            print("Data Baru", data)
            print("*" * 100)
        self.Output(data)

    def Output(self, data):
        fp = Fp.Forward(data)
        for y in range(len(self.Y_output)):
            self.Y_output[y] = fp.sigma(self.w[-(len(self.Y_output) - y)],
                                        self.b[-1][y])
        self.hiddenOutput[self.hidden] = data
        self.softmax = fp.softmax(self.Y_output)
        self.error = fp.error(self.label, self.softmax)
        print("Output", self.o)
        print("Y Output", self.Y_output)
        print("SoftMax", self.softmax)
        print("Error", self.error)
        print("Prediction Class", np.argmax(self.softmax), " - Target Class", np.argmax(self.label))
        print("Hidden Output", self.hiddenOutput)


    def backwardProses(self):
        print("*" * 100, "\nBackward")
        self.outputHidden()
        self.hiddenInput()

    def outputHidden(self):
        count = len(self.softmax)
        for i in range(self.hidden):
            for p in range(self.perc[i]):
                self.eh[i][p] = self.bp.errorHidden(self.w[count + p], self.b[count - 1][p], self.error[p])
            count -= self.perc[i]
        deltaW = self.bp.deltaWeightSoftmax(self.error, self.softmax, self.o, np.argmax(self.label))
        weightNew = self.bp.weightNew(deltaW.reshape(numpy.shape(self.w[-len(self.softmax):])),
                                      self.w[-len(self.softmax):])
        print("Bobot lama", self.w)
        self.w[-2:] = weightNew.tolist()
        print("Delta Weight", weightNew.tolist())
        print("Bobot Baru", self.w)
        deltaBias = self.bp.deltaBiasSoftmax(self.error, self.softmax, np.argmax(self.label))
        print(deltaBias)
        print("Bias Lama", self.b)
        biasNew = self.bp.biasNew(deltaBias, self.b[self.hidden])
        self.b[self.hidden] = biasNew.tolist()
        print("Bias Baru", self.b)

    def hiddenInput(self):
        for i in range(self.hidden):
            for p in range(self.perc[i]):
                deltaWeight = self.bp.deltaWeightSigmoid(self.eh[i][p], self.hiddenOutput[-(i + 1)][p],
                                                         self.hiddenOutput[-(i + 2)])
                deltaBias = self.bp.deltaBiasSigmoid(self.eh[i][p], self.hiddenOutput[-(i + 1)][p])
                self.w[-(len(self.softmax) + self.perc[i]) + p] = self.bp.weightNew(deltaWeight,
                                                                                    self.w[-(len(self.softmax) +
                                                                                             self.perc[i]) + p])
                self.b[(self.hidden-i)-1][p] = self.bp.biasNew(deltaBias, self.b[(self.hidden-i)-1][p])
                print("Delta Weight ", deltaWeight)
                print("Hasil Weight ", self.w[-(len(self.softmax) + self.perc[i]) + p])
                print("Delta Bias", deltaBias)
                print("Hasil Bias", self.b[(self.hidden-i)-1][p])

        print("Weight Baru", self.w)
        print("Bias Baru", self.b)


# dataset = [[1, 1],
#            [5, 4],
#            [2, 6],
#            [7, 2],
#            [3, 2],
#            [7, 5]]

# target = np.array([1, 0, 1, 0, 1, 0])

label = [0, 1]

nn = MLP([1, 1], label, 1, [2])

batasError = 0.1

if nn.valid:
    nn.setWeight()
    iterasi = 0
    while True:
        print("-"*100)
        print("Iterasi ", iterasi)
        nn.forwardProses()
        # break
        if nn.softmax[np.argmax(label)] >= np.max(label)-batasError:
            print(nn.error)
            break
        else:
            print("Masih Error", nn.error)
            nn.backwardProses()
        iterasi += 1
    print("Bobot ", nn.w)
    print("Bias ", nn.b)
