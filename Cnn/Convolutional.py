import numpy as np


class Cnns:
    image = np
    kernel = np
    conImage = np
    normImage = np
    maxPoll = None

    def __init__(self, image, kernel):
        self.image = image
        self.kernel = kernel

    def convolution(self):
        if self.maxPoll is not None:
            self.image = self.maxPoll
            newArr = np.zeros((self.kernel.shape[0], self.image.shape[1] - self.kernel.shape[1] + 1, self.image.shape[2] - self.kernel.shape[2] + 1))
        else:
            newArr = np.zeros((self.kernel.shape[0], self.image.shape[0] - self.kernel.shape[1] + 1, self.image.shape[1] - self.kernel.shape[2] + 1))
        # print("Dimensi Image Baru", self.image.shape[0] - (self.kernel.shape[0] - 1), " x " ,self.image.shape[1] - (self.kernel.shape[0] - 1))
        for i in range(self.kernel.shape[2]):
            for y in range(self.image.shape[1] - (self.kernel.shape[1] - 1)):
                for x in range((self.image.shape[1] - (self.kernel.shape[2] - 1))):
                    newArr[i][y][x] = np.sum(self.image[y:y+3, x:x+3] * self.kernel[i] if len(self.image.shape) <= 2 else self.image[i, y:y+3, x:x+3] * self.kernel[i])
        self.conImage = newArr
        print(newArr.shape)
        return self.conImage

    def ReLU(self):
        self.normImage = np.array([[[x if x >= 0 else 0 for x in i] for i in value] for value in self.conImage])
        print("ReLU\n", self.normImage)
        return self.normImage

    def pooling(self, shape, mode="max", stride=2):
        print(self.normImage)
        inputWidth = self.normImage.shape[2]
        inputHeight = self.normImage.shape[1]
        filterWidth = shape
        filterHeight = shape
        outputWidth = int((inputWidth-filterWidth)/stride)+shape
        outputHeight = int((inputHeight-filterHeight)/stride)+shape
        outputImage = np.zeros((self.normImage.shape[0], outputHeight, outputWidth))

        for i in range(outputImage.shape[0]):
            yOut = 0
            xOut = 0
            for y in range(0, self.normImage.shape[1], stride):
                y_akhir = y+shape if y+shape < self.normImage.shape[1] else y+1
                for x in range(0, self.normImage.shape[2], stride):
                    x_akhir = x+shape if x+shape < self.normImage.shape[2] else x+1
                    if mode == "max":
                        outputImage[i, yOut, xOut] = np.max(self.normImage[i, y:y_akhir, x:x_akhir])
                    elif mode == "average":
                        outputImage[i, yOut, xOut] = np.average(self.normImage[i, y:y_akhir, x:x_akhir])
                    xOut += 1
                yOut += 1
                xOut = 0
        self.maxPoll = outputImage
        print("Max Polling\n", self.maxPoll)

    def stacking(self, feature):
        print("Stacking\n", feature.ravel())


arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]])

arrKernel1 = np.array([[[1, -1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1]],

                       [[1, -1, 1],
                        [-1, 1, -1],
                        [1, -1, 1]],

                       [[-1, -1, 1],
                        [-1, 1, -1],
                        [1, -1, -1]]
                       ])

# print(arr.shape)
convNet = Cnns(arr, arrKernel1)
print(convNet.convolution())
convNet.ReLU()
convNet.pooling(2, mode="max")
print(convNet.convolution())
stack = convNet.ReLU()
convNet.stacking(stack)
