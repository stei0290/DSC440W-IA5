#*********************************************************************
#File name:     A5.py
#Author:        Andrew, Hannah, Roman
#Date:  	    05/08/23
#Class: 	    DSCI 440W
#Assignment:    IA5
#Purpose:       PCA   
#**********************************************************************

#imports
import sympy as sp
import numpy as np
import pylab as pp
import math
import csv
import matplotlib.pyplot as plt
from sympy import *
from numpy.linalg import inv, norm
sp.init_printing(use_unicode=True, use_latex='mathjax')

TRAIN_FILE = "usps-4-9-train.csv"
TEST_FILE = "usps-4-9-test.csv"
#TRAIN_FILE = "knownData.csv"
testLabel_Index = 256
trainLabel_Index = 3

#Read Data
def readCSVData(fileName):
    """    
    Function:   readData
    Descripion: Opens and reads text file
    Input:      fileName - name of file to read from
    Output:     dataList - numpy array of data from file being read
    """
    dataList = []
    # with open(fileName, "r") as f:
    #     raw = f.read()
    #     for line in raw.split("\n"):
    #         subLine = line.split()
    #         dataList.append(subLine)

    with open(fileName, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            # print(row)
            dataList.append(row)
    return dataList

def computeCovariacneMatrix (data):
    """
    Function:       computeCovarianceMatrix
    Description:    computes the covarience matrix for training data
    Input:          data  - matrix of data to find covarience of
    Output:         covMatrix - Covarience matrix of data
    
    """

    covMatrix = []
    centerMat = []
    centerRow = []

    meanArr = computeVectorMean(data)
    trainRows = len(data)
    trainCol = len(data[0])

    for i in range(trainRows):
        for j in range(trainCol):
            center = data[i][j] - meanArr[j]
            centerRow.append(center)
        centerMat.append(centerRow)
        centerRow = []
    centerMat = np.array(centerMat)
    # centermat is the matrix after subtracoin is performed
    # print(centerMat)
    product = np.dot(centerMat,centerMat.T)
    covMatrix = np.divide(product, trainRows)

    return(covMatrix)

def computeVectorMean(matrixX):
    """
    Function:   computeVectorMean
    Descripion: computes the vector mean of a matrix and returns the mean of each column in an array
    Input:      matrixX - matrix of vectors to be compued
    Output:     meansArray - array of vector means
    """
    meansArr = []
    #np.cov(matrix)
    matXT = matrixX.T
    numCols = len(matXT[0])
    numRows = len(matXT)

    for i in range(numRows):
        avg = np.mean(matXT[i])
        meansArr.append(avg)

    return(meansArr)

def chosenEigen (orderedEigenVals, threshold):
    """
    Function:       chosenEigen
    Description:    computes number of eigenValues to use to retain a certain variance threshold
    Input:          orderedEigenVals - ordered eigen values to calculate with
                    threshold  decimal representation of desired variance
    Output:         useValues - the number of eigen values to use 
    
    """
    sum = float(0)
    sumArry = []
    useValues = 0

    for val in orderedEigenVals:
        sum = sum + float(val)
        sumArry.append(sum)

    for num in range (len(orderedEigenVals)):
        if float(sumArry[num]) / float(sum) < threshold:
            useValues = useValues + 1

    return useValues + 1


def driver ():
    testData = []
    trainData = []
    trainOutput = []
    testOutput = []


    testData = readCSVData(TEST_FILE)
    trainData = readCSVData(TRAIN_FILE)

    #Apply outpts to its own array
    for i in range(len(trainData)):
        trainOutput.append(trainData[i][trainLabel_Index])
    for i in range(len(testData)):
        testOutput.append(testData[i][testLabel_Index])

    #Convert outputs to numpy array
    trainOutput = np.array(trainOutput, dtype=float)
    testOutput = np.array(testOutput, dtype=float) 

    #Create input array
    trainFeatures = np.array(trainData, dtype=float)
    testFeatures = np.array(testData,dtype=float)

    #Delte label from feature array
    trainFeatures = np.delete(trainFeatures,trainLabel_Index, axis=1)
    testFeatures = np.delete(testFeatures,testLabel_Index, axis=1)

    numTrainFeatures = len(trainOutput)
        #reshape arary
    trainOutput = trainOutput.reshape(numTrainFeatures,1)
    numTestFeatures =  len(testFeatures)
    testOutput = testOutput.reshape(numTestFeatures,1)
    numTrainFeatures = len(trainFeatures)


    ##help ensure training data is read in properly I lareda checkd andit looks good
    # counter4 = 0
    # counter9 = 0
    # for i in range(numTrainFeatures):
    #      length = (len(trainFeatures[i]))
    #      if (length == 256):
    #         if (trainOutput[i] == 0):
    #             counter4 += 1
    #         elif (trainOutput[i] == 1):
    #              counter9 += 1
    #         else:
    #          print("data read error")
             
    # if (counter9 == counter4 and counter9 == 700):
    #     print("train data read in properly")


    """    
    trainFeatTest = []
    for row in trainFeatures[0:256]:
        trainFeatTest.append(row)
    trainFeatures = np.array(trainFeatTest)
    print(trainFeatures.shape)
    """    
    ## Compute Covarience matix
    
    trainFeatures = np.dot(trainFeatures.T, trainFeatures)
        
    covMatrix = computeCovariacneMatrix(trainFeatures)
    eigenVal, eigenVect = np.linalg.eig(covMatrix)
    # Sort eigen values and vectors from largest to smallest
    idx = eigenVal.argsort()[::-1]
    eigenVal = eigenVal[idx]
    eigenVect = eigenVect[:,idx]

# use .real to get the real parts of the vector
    # print(eigenVect[0])
    # print(type(eigenVect[0]))
    # print(eigenVect[0].real)

    # print(type(eigenVal[0].real))
    # for item in eigenVect:
    #     for num in item:
    #         # num = float(num)
    #         print(type(num))  


    eigenValReal = []
    for item in eigenVal:
        eigenValReal.append(item.real)
    
    
    eigenVectReal = []
    for item in eigenVect:
        vector = []
        for num in item:
            vector.append(num.real)
        eigenVectReal.append(vector)

    # print(type(eigenValReal[0]))
    # print(type(eigenVectReal[0][0]))

    print (chosenEigen (eigenValReal, 0.75))
    
    eigVec = np.array(eigenVectReal)
    displayEigArry = []
    for row in eigVec.T:
        displayEigArry.append(row[0:256])
    

    display = np.array (displayEigArry)
    

    plt.figure(figsize=(16, 16))
    hlp = np.reshape(display[0], (16, 16))
    for item in hlp:
        for num in item:
            num = float(num)
    color_map = plt.imshow(hlp.transpose())
    color_map.set_cmap("Blues_r")


driver()