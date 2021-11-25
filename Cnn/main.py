from NN import MLP
import numpy as np


class main:

    def __init__(self, data, label, hidden, perceptron, batasError=0.1):
        self.data = data
        self.label = label
        self.batas = batasError
        self.hidden = hidden
        self.perceptron = perceptron

    def loopingData(self):
        w = b = []
        neural = MLP(self.hidden, self.perceptron, self.batas)
        if neural.valid:
            for i in range(len(self.data)):
                print("x"*100)
                neural.setData(self.data[i], self.label[i])
                neural.setWeight(w, b)
                neural.forwardProses()
                if neural.softmax[np.argmax(self.label[i])] >= np.max(self.label[i]) - self.batas:
                    print("Error Batas", neural.error)
                else:
                    print("Masih Error", neural.error)
                    neural.backwardProses()
                    neural.hiddenOutput = []
                    neural.o = []
                    w = neural.w
                    b = neural.b
        else:
            print("Jumlah Perceptron dengan Hidden Layer tidak sesuai")



dataset = [[1, 1],
           [3, 2],
           [2, 6],
           [5, 4],
           [7, 2],
           [7, 5]]

target = [[0, 1],
          [0, 1],
          [0, 1],
          [1, 0],
          [1, 0],
          [1, 0]]

# label = [0, 1]

hiddenLayer = 1 #Jumlah Layer Hidden diluar Layer Output
perceptronPerLayer = [2] #berupa List, setiap value mewakili jumlah Perceptron pada setiap layer
#Batas Error Optional Parameter, default 0.1

run = main(dataset, target, hiddenLayer, perceptronPerLayer)
run.loopingData()


