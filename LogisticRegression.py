
import sys
import getopt
import os
import math
import operator
import copy
from collections import defaultdict

SIMPLE_TRAIN = "cs109-datasets-mac/simple-train.txt"
SIMPLE_TEST = "cs109-datasets-mac/simple-test.txt"
HEART_TRAIN = "cs109-datasets-mac/heart-train.txt"
HEART_TEST = "cs109-datasets-mac/heart-test.txt"
VOTE_TRAIN = "cs109-datasets-mac/vote-train.txt"
VOTE_TEST = "cs109-datasets-mac/vote-test.txt"

CASES = [(SIMPLE_TRAIN, SIMPLE_TEST), (VOTE_TRAIN, VOTE_TEST), (HEART_TRAIN, HEART_TEST)]
CASES = [(HEART_TRAIN, HEART_TEST)]

def readFiles(train):
    values = []
    numInput = 0
    numData = 0
    Ys = [0, 0]
    with open(train) as f:
        i = 0
        for line in f:
            if i < 2:
                if i == 0:
                    numInput = int(line)
                else:
                    numData = int(line)
            elif i < numData + 2:
                vals = []
                Y = 0
                if i == numData + 1:
                    Y = int(line[-1])
                else:
                    Y = int(line[-2])
                Ys[Y] += 1
                for index in range(numInput):
                    vals.append(int(line[index * 2]))
                tempArray = []
                tempArray.append(tuple(vals))
                tempArray.append(Y)
                values.append(tempArray)
            i += 1
    return (values, numInput, numData, Ys)


def logReg(values, numInput, epochs, learningRate):
    betas = [0] * numInput
    for i in range(epochs):
        gradient = [0] * numInput
        for train in values:
            z = 0
            for j in range(numInput):
                z += betas[j] * train[0][j]
            for j in range(numInput):
                gradient[j] += (train[0][j]) * (train[1] - 1 / (1 + math.e ** (-z) ) )
        for j in range(numInput):
            betas[j] += learningRate * gradient[j]
    #print("betas", betas)
    return betas


def makePrediction(testValues, betas, numInput):
    correct = 0
    wrong = 0
    myAnswers = []
    for test in testValues:
        answer = test[1]
        z = 0
        for j in range(numInput):
            z += betas[j] * test[0][j]
        prob1 = 1 / (1 +  math.e ** (-z))
        if prob1 > 0.5:
            myAnswers.append(1)
        else:
            myAnswers.append(0)
        if myAnswers[-1] == answer:
            correct += 1
        else:
            wrong += 1
    print("\tAccuracy:", correct / (correct + wrong + 0.0))


for case in CASES:
    train = case[0]
    test = case[1]
    epochs = 10000
    learningRate = 0.000000000000000000010164395367051604
    print("learningRate", learningRate)
    values, numInput, numData, Ys = readFiles(train)
    betas = logReg(values, numInput, epochs, learningRate)
    testValues, testNumInput, testNumData, testYs = readFiles(test)
    makePrediction(testValues, betas, numInput)
