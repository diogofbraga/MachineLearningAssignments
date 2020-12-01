import pandas as pd
import numpy as np 
import pylab as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
'''
Calculate linear regression, using specified loss function.
BEWARE: using an exact solution leads to division by zero because the RMSE is zero. Should be solved somehow... 
        but should not be happening with real data anyways...


linearRegression(X, y, lossFunction = 'RMSE', alpha = 1, convergenceCriterion = 1e-4, adaptAlpha = False, printOutput = False):

Input Parameters:
1) X and y are analogous in use to scikit learn use. But MUST be pandas arrays here
2) loss Function is used to calculate the deviation from the prediction to the target variable
3) alpha is the leraning rate
4) convergenceCriterion sets the target accuracy for the fit. If the result of the loss function is smaller than 
   convergenceCriterion then the program stops.
5) adaptAlpha divides alpha by 10, if loss function result is smaller than 10*alpha <- much better approaches can probably be found
6) printOutput True prints important information from every loop. (loss function result, weights, initial weights, alpha)
7) initialWeigts, if set to random weights are initialized randomly, if set to manual, then an array with the weights according to:
        Y = w1*X1 + w2*X2 + ... + wB*Xb <- here Xb is the bias column (=[1]*NVal)
   needs to be specified.
'''
#using functions for loss function and derivative for now because it is more easily readable that way I think
#probably less performant in python though...?
def RMSE(y, yPred):
    return(float(np.sqrt(y.subtract(yPred, axis = 0).pow(2).mean())))
    #return(np.sqrt(np.mean(np.square(y - yPred))))

def ddwRMSE(i, X,  y, yPred, weights, RMSE):
    return(float(np.mean(y.subtract(yPred, axis = 0).multiply(-X.iloc[:,i], axis=0))) / (RMSE))

def updateWeights(X, y, yPred, weightsOld, lossFunction, alpha):
    NWeights = len(weightsOld)
    weightsNew = np.zeros(NWeights)
    if lossFunction == 'RMSE':
        loss = RMSE(y, yPred)
        for i in range(0, NWeights):
            weightsNew[i] = weightsOld[i] - alpha * ddwRMSE(i, X, y, yPred, weightsOld, loss)
    else:
        print('ERROR: lossFunction = ', lossFunction, ' not found!')
        loss = 0
        weightsNew = weightsOld    
    
    return(weightsNew, loss)

def linearRegression(X, y, lossFunction = 'RMSE', alpha = 1, convergenceCriterion = 1e-4, adaptAlpha = False, printOutput = False, initialWeights = 'Random'):
    NVar = len(X.columns)
    XCount = X.count()   

    #assuming all columns have been preprocessed to have the same number of values!
    #check if all columns are of same length, error otherwise:
    lenCols = []
    for i in range(0, NVar):
        lenCols.append(XCount[i])
    if all(x == lenCols[0] for x in lenCols) == True:
        NVal = X.count()[0]
    else:
        print('ERROR: Columns have different number of values!')
        return()
    
    #because it is easier to do the derivative later by not needing an exception for the bias weight 
    #I insert a first column with just ones at the beginning of the dataframe that is then multiplied 
    #with the bias weight
    biasCol = np.ones(NVal)
    X.insert(loc = NVar, column = 'bias', value = biasCol)
    NVar += 1

    #weigths for each column plus bias
    if initialWeights == 'Random':
        weights = np.random.rand(NVar)
    if initialWeights == 'Manual':
        weights = initialWeights
    #weights = np.array([3.00001,1.00001,2.00001])
    if printOutput == True:
        print('Initial weights = ', weights)    
        
    loss = convergenceCriterion + 1
    counter = 0
    while loss > convergenceCriterion:
        if adaptAlpha == True:
            if alpha > loss/10:
                alpha /= 10
        
        #eqaution to be solved:
        #Y = w0*X0 + w1*X1 + w2*X2 ...<- here X0 is the bias column (=[1]*NVal)
        yPred = X.multiply(weights, axis = 1).sum(axis = 1)
        
        #update weights and calc loss function and its derivative:
        weights, loss = updateWeights(X, y, yPred, weights, lossFunction, alpha)
        if printOutput == True:
            print('===================================')
            print('iteration = ', counter)
            print('loss =', loss)
            print('alpha =', alpha)
            print('updated weights = ', weights)    
        counter += 1
    return()


if __name__ == '__main__':
    data = pd.read_csv('../MetroInterstateTrafficVolumeTest.csv', delimiter = ',')
    
    #make it so that number of columns doesn't matter and deal with one hot encoding later
    #X = data.drop(['traffic_volume', 'holiday', 'weather_main', 'weather_description', 'date_time'], axis=1)
    #y = data['traffic_volume']
    X = pd.DataFrame(data = np.array([[1, 1], [1, 2], [2, 2], [2, 3]]))
    y = pd.DataFrame(data = (np.dot(X, np.array([1, 2])) + 3))
    print(X)
    print(y)
    
    linearRegression(X, y, alpha = 1, convergenceCriterion = 1e-5, adaptAlpha = True, printOutput = True)
    
    print('==================================================')
    X = pd.DataFrame(data = np.array([[1, 1], [1, 2], [2, 2], [2, 3]]))
    y = pd.DataFrame(data = (np.dot(X, np.array([1, 2])) + 3))
    
    reg = LinearRegression(normalize = False).fit(X, y)
    print('reg.score(X,y) =', reg.score(X, y))
    print('reg.coef_ = ', reg.coef_)
    print('reg.intercept_ = ',reg.intercept_)
    



