# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:14:55 2018

@author: praveen joshi
"""
#######################################Libraries to import#####################################
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.grid_search import ParameterGrid
import time

#######################################Import data from file####################################
def getData(filePath):
    """
    Reads file from filePath and save it as ndarray.
    :param filePath: takes in file path to be loaded
    :return: data in ndarray
    """
    _data = genfromtxt(filePath, delimiter=',')
    return _data

#######################################Distance Functions#######################################
def calculateDistances(compArray, queryArray):
    """
    Computes Euclidean distance-
    :param compArray: takes in 2d array i.e. training dataset 
    :param queryArray: takes in 1d array i.e. instance of test dataset
    :return: sqrtArray :distance between the querry point and each point in training dataset
             sortedIndex : sorted index based on the distance between the querry point and each point in training dataset
    """
    subArray =np.subtract(compArray,queryArray)
    powArray = subArray**2
    sumArray = np.sum(powArray,axis =1)
    sqrtArray = np.sqrt(sumArray)
    #sorts index of array based on respective location value
    sortedIndex = np.argsort(sqrtArray)
    return sqrtArray,sortedIndex    

def calculateManhattanDistances(compArray, queryArray):
    """
    Computes Manhattan distance-
    :param compArray: takes in 2d array i.e. training dataset 
    :param queryArray: takes in 1d array i.e. instance of test dataset
    :return: sumArray :distance between the querry point and each point in training dataset
             sortedIndex : sorted index based on the distance between the querry point and each point in training dataset
    """
    subArray =np.subtract(compArray,queryArray)
    subArray= np.absolute(subArray)
    sumArray = np.sum(subArray,axis =1)
    #sorts index of array based on respective location value
    sortedIndex = np.argsort(sumArray)
    return sumArray,sortedIndex  

def calculateMinkowskiDistances(compArray, queryArray, alpha):
    """
    Computes Euclidean distance-
    :param compArray: takes in 2d array i.e. training dataset 
    :param queryArray: takes in 1d array i.e. instance of test dataset
    :param alpha: this value allow us to play with multiple values in which 1 =manhattan distance and 2= euclidean distance
    :return: sqrtArray :distance between the querry point and each point in training dataset
             sortedIndex : sorted index based on the distance between the querry point and each point in training dataset
    """
    subArray =np.subtract(compArray,queryArray)
    np.absolute(subArray)
    powArray = subArray**alpha
    sumArray = np.sum(powArray,axis =1)
    sqrtArray = np.power(sumArray, (1./float(alpha)))
    #sorts index of array based on respective location value
    sortedIndex = np.argsort(sqrtArray)
    return sqrtArray,sortedIndex

#######################################Splitting test and train data##############################
def splitData(array):
    """
    Splits Data into feature dataset and target dataset-
    :param array: takes in 2d array
    :return: array1 : which contains only features
             array2 : contains only target variable
    Note: It will only work when target variable is present in end.
    """
    return array[:, :-1], array[:, -1]

#######################################Get prediction function####################################
def getPrediction(computedList, Ytrain, k):
    """
    This functions helps in generating prediction list-
    :param computedList: takes in list in which each element contains distance of that respective test point to all training datapoint
    :param Ytrain: takes in 1d array, it contains the ground truth of training dataset
    :param k: It is hyper parameter which decides how many nearest neighbour should be kept in consideration while doing prediction.
    :return: predictionList : which contains predictions based on the given input
    """
    predictionList=[]
    for sortedIndex in computedList:
        zeroVote = 0
        for count in range(0,k):
            if Ytrain[sortedIndex[count]] == 0:
                zeroVote = zeroVote +1
        #if Vote for class 0 is greater than the half of the voting guys decided by parameter k 
        #prediction is given as class0
        if zeroVote >= (int(k/2)+1):
            predictionList.append(0)
        else:
            predictionList.append(1)
    return predictionList

def getWtPrediction(computedListValue, k, n):   
    """
    This functions helps in generating prediction for each test data point-
    :param computedListValue: takes in list in which distance of that respective test point to all training datapoint are available
    :param k: It is hyper parameter which decides how many nearest neighbour should be kept in consideration while doing prediction.
    :param n: It is another hyper-parameter which decides the power to the inverse distance.
    :return: ndComputedList : which contains prediction based on the given input
    """
    ndComputedList= np.array(computedListValue)
    ndComputedList.sort(axis=1)
    ndComputedList= ndComputedList[:, :k]
    ndComputedList = 1/pow(ndComputedList,n)
    ndComputedList = np.sum(ndComputedList, axis =1) 
    return ndComputedList

#######################################Plot and save graph#######################################
def drawPlot(predictionWithkTune,titel,xAxis,yAxis):
    """
    This functions helps in plotting the scatter graph-
    :param predictionWithkTune: Data for scatter plot
    :param titel: Title for scatter plot
    :param xAxis: x Axis label for scatter plot
    :param yAxis: yAxis label for scatter plot
    :return: none
    """
    plt.scatter(*zip(*predictionWithkTune))
    plt.title(titel)
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    #plot will be saved as 'config' + timestamp as its file number
    fileName = 'config',time.strftime("%H_%M_%S", time.gmtime()),'.png'
    plt.savefig(''.join(fileName))
    plt.show()
    
#######################################Functions for Regression Problem##########################
    
def normaliseData(data):
    """
    This functions helps in normalising the data-
    :param data: Data is normalised by subtracting the oldValue with mininmum value of that feature devided by range of that feature
    :return: newValue: normalised value
    """
    minData =  np.amin(data, axis=0)
    maxData = np.amax(data, axis=0)
    #computation of range for each column
    denominator = maxData-minData
    #computing newValue by subtracting oldValue with the minValue present for that feature
    newValue = data-minData
    #final newValue will generated after dividing the above stage newValue with the range of value for that particular feature.
    newValue = newValue/denominator
    return newValue

def getPredictionRegression(computedListDist, YtrainR,k):
    """
    This functions helps in generating prediction list-
    :param computedList: takes in list in which each element contains distance of that respective test point to all training datapoint
    :param Ytrain: takes in 1d array, it contains the ground truth of training dataset
    :param k: It is hyper parameter which decides how many nearest neighbour should be kept in consideration while doing prediction.
    :return: predictionList : which contains predictions based on the given input
    """
    predictionList =[]
    for distListItem in computedListDist:
        ndComputedList= np.array(distListItem)
        #if any datapoint is found to distance 0 it's been replaced by 0.1(least distance from datapoint) to avoid any mathematical error
        #this parameter in future can be removed or tuned based on the performance
        ndComputedList[ndComputedList == 0.] = 0.1
        ndComputedListIndex = np.argsort(ndComputedList)
        ndComputedListIndex = ndComputedListIndex[:k]
        ndComputedList.sort()
        ndComputedList= ndComputedList[:k]
        ndComputedList = pow(ndComputedList,2)   
        temp = 1/ ndComputedList
        denom = np.sum(temp)
        count =0
        sumNum=0
        #weighted sum is evaluated underneath-
        while count<k:
            sumNum = sumNum +((YtrainR[ndComputedListIndex[count]])* (float(1./ndComputedList[count])))
            count = count +1
        value =sumNum / denom
        predictionList.append(value)
    return predictionList

def getSumOfSquaredResiduals(yhat,y):
    """
    This functions helps in computing sum of squared residuals-
    :param yhat: this is models prediction
    :param y: this is ground truth
    :return: sumOfSquaredResiduals : this is sum of squared residuals
    """
    temp= np.subtract(yhat,y)
    temp = np.square(temp)
    return np.sum(temp)
    
def getTotalSumOfSquares(y):
    """
    This functions helps in computing total sum of squares-
    :param y: this is ground truth
    :return: totalSumOfSquares : this is total Sum Of Squares
    """
    ybar = np.mean(y)
    temp= np.subtract(ybar,y)
    temp = np.square(temp)
    return np.sum(temp)
       
def getRsquare(yhat, y):
    """
    This functions helps in computing Rsquare value-
    :param yhat: this is models prediction
    :param y: this is ground truth
    :return: sumOfSquaredResiduals : this is Rsquare value
    """
    #computation of sum of squared residuals
    sumOfSquaredResiduals = getSumOfSquaredResiduals(yhat,y)
    #computation of total sum of squares
    totalSumOfSquares = getTotalSumOfSquares(y)
    counterRsquare = float (sumOfSquaredResiduals / totalSumOfSquares)
    #returns Rsquare value
    return 1-counterRsquare


#################################################################################################################
######################################Main Function##############################################################
#################################################################################################################
if __name__ == "__main__":
    
#################################################################################################################    
#####################################Development of k-NN Algorithm######################################
#################################################################################################################    
    """
    This section takes in data and split the data accordingly
    """
    test = 'testData2.csv'
    train = 'trainingData2.csv'
    
    testData = getData(test)
    trainData = getData(train)
    
    Xtest, Ytest = splitData(testData)
    Xtrain, Ytrain = splitData(trainData)
    
#Uncomment underneath lines for normalised dataset creation
#    Xtest = normaliseData(Xtest)
#    Xtrain = normaliseData(Xtrain)
    
    
#################################################Normal Euclidean Distance###################################
    """
    Best value of k is determined based on the accuracy.
    This section makes use of calculateDistances to compute distance from each test datapoint to all training data point.
    """
    computedList =[]
    
    #distance evaluation for each test datapoint against the traing datapoints
    for test in Xtest:
        sortedDist,sortedIndex = calculateDistances(Xtrain,test)
        computedList.append(sortedIndex)
    
    #list to obtain series for k value against accuracy
    predictionWithkTune =[]
    #best accuracy value while evaluation
    bestKValue =0 
    #best k value index while evaluation
    bestKValueIndex =0 
    #changing hyper-parameter k from 1 to 100
    for kCount in range(1,100):    
        predictionList = getPrediction(computedList, Ytrain,kCount)
        
        #computation of accuracy
        accuracy = predictionList == Ytest
        sumAccuracy = np.sum(accuracy)
        perAccuracy = (sumAccuracy/len(Ytest))*100
        
        #finding accuracy for k=3 for part 1
        if kCount == 3:
            print('for k =3 accuracy', perAccuracy)
        
        #keeping track of best k-value and accuracy
        if perAccuracy > bestKValue:
            bestKValue = perAccuracy
            bestKValueIndex = kCount
        predictionWithkTune.append((kCount,perAccuracy))
    #printing tuned hyper-parameter and accuracy
    print('best k value found at: ',bestKValueIndex,' accuracy obtained: ', bestKValue)
    #plotting plot of k-value against accuracy
    drawPlot(predictionWithkTune,'k-Value V/S Accuracy','k-value','accuracy')
 
#################################################################################################################    
#############################Investigating kNN variants and hyper-parameters############################
#################################################################################################################    
#################################################Manhattan Distance##############################################
    """
    Thre more distance metrics are introduced and analysis of hyper parameter is done
    Underneath is implementation of Manhattan metric for prediction.
    """
    computedList =[]
    
    #distance evaluation for each test datapoint against the traing datapoints
    for test in Xtest:
        sortedDist,sortedIndex = calculateManhattanDistances(Xtrain,test)
        computedList.append(sortedIndex)
        
    #list to obtain series for k value against accuracy
    predictionWithkTune =[]
    #best accuracy value while evaluation
    bestKValue =0 
    #best k value index while evaluation
    bestKValueIndex =0 
    #changing hyper-parameter k from 1 to 100
    for kCount in range(1,100):    
        predictionList = getPrediction(computedList, Ytrain,kCount)
        
        #computation of accuracy
        accuracy = predictionList == Ytest
        sumAccuracy = np.sum(accuracy)
        perAccuracy = (sumAccuracy/len(Ytest))*100
    
        #keeping track of best k-value and accuracy
        if perAccuracy > bestKValue:
            bestKValue = perAccuracy
            bestKValueIndex = kCount
        predictionWithkTune.append((kCount,perAccuracy))
    #printing tuned hyper-parameter and accuracy        
    print('best k value found at: ',bestKValueIndex,' accuracy obtained: ', bestKValue)
    #plotting plot of k-value against accuracy
    drawPlot(predictionWithkTune,'k-Value V/S Accuracy','k-value','accuracy')
    
#################################################Minkowski Distance#########################################    
    """
    Underneath is implementation of Minkowski metric for prediction.
    """
    #list to obtain series for k value against accuracy
    predictionWithkMDTune =[]
    #list to obtain series for alpha value against accuracy
    predictionWithalphaMDTune =[]
    #best accuracy value while evaluation
    bestKValue =0 
    #best k value index while evaluation
    bestKValueIndex =0 
    #best alpha value index while evaluation
    bestalphaValueIndex =0
    
    #changing hyper-parameter k from 1 to 100
    #changing hyper-parameter alpha from list - .1,.2,.3,.4,.5,.6,.7,.8,.9,1,2,3,4,5,6,7,8,9,10
    param_grid = {'param1': range (1, 100), 'paramN' : [.1,.2,.3,.4,.5,.6,.7,.8,.9,1,2,3,4,5,6,7,8,9,10]}
    #formation of grid matrix for further evaluation
    grid = ParameterGrid(param_grid)
    
    for params in grid:
        
        computedList =[]
        
        #distance evaluation for each test datapoint against the traing datapoints
        for test in Xtest:
            sortedDist,sortedIndex = calculateMinkowskiDistances(Xtrain,test,params['paramN'])
            computedList.append(sortedIndex)
            
        predictionWithkTune =[]
        
        predictionList=[]
        predictionList = getPrediction(computedList, Ytrain,params['param1'])
            
        #computation of accuracy
        accuracy = predictionList == Ytest
        sumAccuracy = np.sum(accuracy)
        perAccuracy = (sumAccuracy/len(Ytest))*100
        
        #keeping track of best k-value,alpha and accuracy
        if perAccuracy > bestKValue:
            bestKValue = perAccuracy
            bestKValueIndex = params['param1']
            bestalphaValueIndex= params['paramN']
        predictionWithkMDTune.append((params['param1'],perAccuracy))
        predictionWithalphaMDTune.append((params['paramN'],perAccuracy))
    #printing tuned hyper-parameters and accuracy        
    print('best k value found at: ',bestKValueIndex,'best alpha value found at: ',bestalphaValueIndex,' accuracy obtained: ', bestKValue)
    #plotting plot of k-value against accuracy
    drawPlot(predictionWithkMDTune,'k-Value V/S Accuracy','k-value','accuracy')
    #plotting plot of alpha-value against accuracy
    drawPlot(predictionWithalphaMDTune,'alpha-Value V/S Accuracy','alpha-value','accuracy')

##################################################Weighted Euclidean Distance###################################### 
    """
    Underneath is implementation of Weighted Euclidean metric for prediction.
    """
    Xtest0, Ytest0 = splitData(testData[testData[:,5]==0])
    Xtrain0, Ytrain0 = splitData(trainData[trainData[:,5]==0])
    
    Xtest1, Ytest1 = splitData(testData[testData[:,5]==1])
    Xtrain1, Ytrain1 = splitData(trainData[trainData[:,5]==1])
    
    computedList0Index =[]
    computedList1Index =[]
    
    computedList0Value =[]
    computedList1Value =[]
    
    #distance evaluation for each test datapoint against the traing datapoints
    for test in Xtest:
        #computation of distance for class 0 
        sortedDist,sortedIndex = calculateDistances(Xtrain0,test)
        computedList0Index.append(sortedIndex)
        computedList0Value.append(sortedDist)
        
        #computation of distance for class 1
        sortedDist,sortedIndex = calculateDistances(Xtrain1,test)
        computedList1Index.append(sortedIndex)
        computedList1Value.append(sortedDist)
        
    #list to obtain series for k value against accuracy
    predictionWithkWtTune =[]
    #list to obtain series for n value against accuracy
    predictionWithNWtTune =[]
    #best accuracy value while evaluation
    bestKValue =0 
    #best k value index while evaluation
    bestKValueIndex =0 
    #best N value index while evaluation
    bestNValueIndex =0
    #changing hyper-parameter k from 1 to 100
    #changing hyper-parameter N-value from 1 to 50
    param_grid = {'param1': range (1, 100), 'paramN' : range (1, 50)}
    #formation of grid matrix for further evaluation
    grid = ParameterGrid(param_grid)
    
    for params in grid:
        predictionList0 =getWtPrediction(computedList0Value, k= params['param1'], n=params['paramN'])
        predictionList1 =getWtPrediction(computedList1Value,  k= params['param1'], n=params['paramN'])
        
        #It will give False whenever its 0
        predictClass = predictionList0<predictionList1 
        predictClass = predictClass.astype(int) 
        
        #computation of accuracy
        accuracy = predictClass == Ytest
        sumAccuracy = np.sum(accuracy)
        perAccuracy = (sumAccuracy/len(Ytest))*100
        
        #keeping track of best k-value,n-value and accuracy
        if perAccuracy > bestKValue:
                bestKValue = perAccuracy
                bestKValueIndex = params['param1']
                bestNValueIndex = params['paramN']
        
        predictionWithkWtTune.append((params['param1'],perAccuracy))     
        predictionWithNWtTune.append((params['paramN'],perAccuracy))    
    #printing tuned hyper-parameters and accuracy  
    print('best k value found at: ',bestKValueIndex,'best n value found at: ',bestKValueIndex,' accuracy obtained: ', bestKValue)
    #plotting plot of k-value against accuracy
    drawPlot(predictionWithkWtTune,'k-Value V/S Accuracy','k-value','accuracy')
    #plotting plot of N-value against accuracy
    drawPlot(predictionWithNWtTune,'N-Value V/S Accuracy','n-value','accuracy')
    
