
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

# values = []
# numInput = 0
# numData = 0
# sums = [[],[]]

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




def convertToSums(values, numInput):
    sums = [[0, 0], [0, 0]]
    allSums = []
    for i in range(numInput):
        allSums.append(copy.deepcopy(sums))
    for arr in values:
        Y = 0
        if arr[1] == 1:
            Y = 1
        for j in range(numInput):
            if arr[0][j] == 0:
                allSums[j][Y][0] += 1
            else:
                allSums[j][Y][1] += 1
    return allSums


def convertToProbability(allSums, numData, Ys):
    for j in range(len(allSums)):
        for y in range(2): #range(len(allSums[j]))
            for b in range(2): #range(len(allSums[j][y]))
                allSums[j][y][b] = allSums[j][y][b] / (numData + 0.0)
    total = Ys[0] + Ys[1]
    Ys[0] = Ys[0] / total
    Ys[1] = Ys[1] / total
    return allSums, Ys


def laplace(allSums, numData, Ys):
    for j in range(len(allSums)):
        for y in range(2): #range(len(allSums[j]))
            for b in range(2): #range(len(allSums[j][y]))
                allSums[j][y][b] = allSums[j][y][b] + 1
    Ys[0] += 2
    Ys[1] += 2
    return allSums, numData + 4, Ys

def makePrediction(allProbs, YsProb, testValues):
    correct = 0
    wrong = 0
    myAnswers = []
    for value in testValues:
        probY0 = 0
        probY1 = 0
        inputs = value[0]
        answer = value[1]
        for i in range(len(inputs)):
            val = inputs[i]
            if i == 0:
                probY0 = allProbs[i][0][val] / YsProb[0]
                probY1 = allProbs[i][1][val] / YsProb[1]
            else:
                probY0 *= allProbs[i][0][val] / YsProb[0]
                probY1 *= allProbs[i][1][val] / YsProb[1]
        if probY0 > probY1:
            myAnswers.append(0)
        else:
            myAnswers.append(1)
        if myAnswers[-1] == answer:
            correct += 1
        else:
            wrong += 1
    print("Accuracy:", correct / (correct + wrong + 0.0))


for case in CASES:
    train = case[0]
    test = case[1]
    values, numInput, numData, Ys = readFiles(train)
    allSums = convertToSums(values, numInput)
    # allSums, numData, Ys = laplace(allSums, numData, Ys)
    allProbs, YsProb = convertToProbability(allSums, numData, Ys)
    testValues, testNumInput, testNumData, testYs = readFiles(test)
    makePrediction(allProbs, YsProb, testValues)
