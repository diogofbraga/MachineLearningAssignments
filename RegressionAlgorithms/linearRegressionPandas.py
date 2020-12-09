import pandas as pd
import numpy as np 
import pylab as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import time
import scipy as scipy
'''
Calculate linear regression, using specified loss function.
BEWARE: using an exact solution leads to division by zero because the RMSE is zero. Should be solved somehow... 
        but should not be happening with real data anyways...


linearRegression(X, y, lossFunction = 'RMSE', alpha = 1, convergenceCriterion = 1e-4, adaptAlpha = False, printOutput = False, maxIterations = None):

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
8) maxIterations sets the maximum number of iterations for the regression loop
'''
#using functions for loss function and derivative for now because it is more easily readable that way I think
#probably less performant in python though...?
def Rsquare(y, yPred):
    return(1 - float(y.subtract(yPred, axis = 0).pow(2).sum() / y.subtract(y.mean(), axis = 0).pow(2).mean()))

#this is dumb... the whole using of functions instead of ifs 
#... probably worse than an if statement in the core iteration?
#this way i feel like it is super readble though and I can still change it if performance is a big issue
#for performance reasons it might also be better to transform everything to numpy arrays?
def alphaConst(alpha0, counter, mu):
    return(alpha0)

def alphaLin(alpha0, counter, mu):
    return(alpha0/(counter*mu))

def alphaSqrt(alpha0, counter, mu):
    return(alpha0/np.sqrt(counter*mu))

def RMSE(y, yPred):
    return(float(np.sqrt(y.subtract(yPred, axis = 0).pow(2).mean())))

def ddwRMSE(i, X,  y, yPred, weights):
    return(float(np.mean(y.subtract(yPred, axis = 0).multiply(-X.iloc[:,i], axis=0))) / (RMSE(y,yPred)))

def MSE(y, yPred):
    return(float(y.subtract(yPred, axis = 0).pow(2).mean()))

def ddwMSE(i, X,  y, yPred, weights):
    return(2*float(np.mean(y.subtract(yPred, axis = 0).multiply(-X.iloc[:,i], axis=0))))

def MAE(y, yPred):
    return(float(y.subtract(yPred, axis = 0).abs().mean()))

def ddwMAE(i, X,  y, yPred, weights):
    return(float(np.mean(np.sign(y.subtract(yPred, axis = 0)).multiply(-X.iloc[:,i], axis=0))))

#weights, loss, alpha = updateWeights(X, y, yPred, weights, alpha, lossDict[lossFunction], ddwLossDict[lossFunction])
#def updateWeights(X, y, yPred, weightsOld, alpha, lossFkt, ddwLossFkt):
#    NWeights = len(weightsOld)
#    weightsNew = np.zeros(NWeights)
#    lossOld = lossFkt(y, yPred)
#    for i in range(0, NWeights):
#        weightsNew[i] = weightsOld[i] - alpha[i] * ddwLossFkt(i, X, y, yPred, weightsOld)
#    loss = lossFkt(y, yPred)
#    return(weightsNew, loss, alpha)

def linearRegression(X, y, lossFunction = 'RMSE', alpha = [], mu = 1, convergenceCriterion = 1e-4, alphaMethod = 'lin', 
                     printOutput = False, initialWeights = 'Zero', maxIterations = None):
    
    lossDict = {
        "RMSE": RMSE,
        "MSE": MSE,
        "MAE": MAE
    }
    
    ddwLossDict = {
        "RMSE": ddwRMSE,
        "MSE": ddwMSE,
        "MAE": ddwMAE
    }
    #at the moment only alphaMethod=const is needed... everything else is deactivated for performance
    alphaMethodDict = {
        "const" : alphaConst,
        "lin" : alphaLin,
        "sqrt" : alphaSqrt
    }

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
        print('ERROR: Columns have different numbers of values!')
        return()
    
    #because it is easier to do the derivative later by not needing an exception for the bias weight 
    #I insert a first column with just ones at the beginning of the dataframe that is then multiplied 
    #with the bias weight
    biasCol = np.ones(NVal)
    X.insert(loc = NVar, column = 'bias', value = biasCol)
    NVar += 1

    if alpha == []:
        alpha = np.ones(NVar)*1000
    alpha0 = alpha

    #weigths for each column plus bias
    if initialWeights == 'Zero':
        weights = np.zeros(NVar)
    if initialWeights == 'Random':
        weights = np.random.rand(NVar)
    if initialWeights == 'Manual':
        weights = np.asarray(initialWeights)
    #weights = np.array([0.223, 0.892, 3.347, 4.661])
    weightsNew = np.zeros(NVar)
    if printOutput == True:
        print('Initial weights = ', weights)    
        
    #eqaution to be solved:
    #Y = w0*X0 + w1*X1 + w2*X2 ...<- here X0 is the bias column (=[1]*NVal)
    yPred = X.multiply(weights, axis = 1).sum(axis = 1)
    lossOld = 0
    lossNew = 10000
    counter = 1
    while np.abs(lossOld-lossNew) > convergenceCriterion:
        if maxIterations == counter:
            break
        
        #alpha = alphaMethodDict[alphaMethod](alpha0, counter, mu)
        lossOld = lossDict[lossFunction](y, yPred) 
        #update weights and calc loss function and its derivative:
        for i in range(0, NVar):
            derivative = ddwLossDict[lossFunction](i, X, y, yPred, weights)
            weightsNew[i] = weights[i] - alpha[i] * derivative
            yPred = X.multiply(weightsNew, axis = 1).sum(axis = 1)
            loss = lossDict[lossFunction](y, yPred)
            #if alpha is too large the error explodes...
            #this makes the solution basially independent of the initial alpha
            #after the while iteration is done for the first time
            #actually initial alpha needs to be large enough!
            while loss > lossOld:
                weightsNew[i] = weights[i] - alpha[i] * derivative
                yPred = X.multiply(weightsNew, axis = 1).sum(axis = 1)
                loss = lossDict[lossFunction](y, yPred)
                alpha[i] /= 1.5

        weights[:] = weightsNew[:]
        yPred = X.multiply(weights, axis = 1).sum(axis = 1)
        lossNew = loss
        
        if printOutput == True:
            print('===================================')
            print('iteration = ', counter)
            print('loss =', loss)
            print('alpha =', alpha)
            print('updated weights = ', weights)
            #print('y =', y)
            #print('yPred =', yPred)
        counter += 1
    #remove the additional bias column again!
    del X['bias']
    return(weights, Rsquare(y, yPred))

def predictLinearRegression(X, weights):
    NVal = X.count()[0]
    NVar = len(X.columns)
    biasCol = np.ones(NVal)
    X.insert(loc = NVar, column = 'bias', value = biasCol)
    yPred = X.multiply(weights, axis = 1).sum(axis = 1)
    #remove the additional bias column again!
    del X['bias']
    return(yPred)

def matrixSolution(X, y):
    #matrix equation that minimizes the sum of squares (Soq)
    # ->dSoq/dw_i = 0
    NVar = len(X.columns)
    NVal = X.count()[0]
    biasCol = np.ones(NVal)
    X.insert(loc = NVar, column = 'bias', value = biasCol)
    NVar += 1 
    dS = np.zeros(NVar)
    wCoeffs = np.zeros((NVar,NVar)) 
    #w = ?
    for i in range(0, NVar):
        dS[i] = float(2*y.multiply(X.iloc[:,i], axis=0).sum(axis=0))
    for i in range(0, NVar):
        for j in range(0, NVar):
            wCoeffs[i,j] = 2*float(X.iloc[:,i].multiply(X.iloc[:,j]).sum(axis = 0)) 
    
    weights = scipy.linalg.solve(wCoeffs, dS) #<- uses LU decomposition therefore faster than inverting matrix...
    
    #remove the additional bias column again!
    del X['bias']
    return(weights)

if __name__ == '__main__':
    path = '../MetroInterstateTrafficVolume/MetroInterstateTrafficVolume.csv'
    data = pd.read_csv(path, delimiter = ',')
    
    #X = data.drop(['rain_1h', 'snow_1h', 'traffic_volume', 'holiday', 'weather_main', 'weather_description', 'date_time'], axis=1)
    #y = data['traffic_volume']
    #X = pd.DataFrame(data = np.array([[1, 1], [1, 2], [2, 2], [2, 3]]))
    #y = pd.DataFrame(data = (np.dot(X, np.array([1, 2])) + 3))
    N1 = 20000000
    x = np.linspace(0,200, num = N1)
    x1 = x+2
    x2 = x*4-3
    #x3 = x*15-7
    N2 = 2
    #N2 = 3
    xges = np.zeros((N1,N2))
    xges[:, 0] = x1[:]
    xges[:, 1] = x2[:]
    #xges[:, 2] = x3[:]
    X = pd.DataFrame(data = xges)
    #y = pd.DataFrame(data = (np.dot(X, np.array([2, 8, 2])) + 4))
    y = pd.DataFrame(data = (np.dot(X, np.array([2, -3])) + 4))
    
    print(X)
    print(y)
    
    print('==================================================')
    
    start = time.time()
    reg = LinearRegression(normalize = False).fit(X, y)
    print('reg.score(X,y) =', reg.score(X, y))
    print('reg.coef_ = ', reg.coef_)
    print('reg.intercept_ = ',reg.intercept_)
    end = time.time()
    print('time = ', end - start)
    
    print('==================================================')
    
    start = time.time()
    #alpha = [1e-9, 1e-9, 1e-9, 2.5e-6]
    #alpha = [5e-4, 5e-2]
    alpha = [500, 500, 500, 500]
    alphaMethod = 'const'
    #mu = 1e-2
    mu = 1
    printOutput = False
    weights, score = linearRegression(X, y, alpha = alpha, mu = mu, convergenceCriterion = 1e-9, 
                                      lossFunction = 'MSE', alphaMethod = alphaMethod, printOutput = printOutput)
    yPred = predictLinearRegression(X, weights)
    #print('yPred = ', yPred)
    print('weights = ', weights)
    print('score = ', score)
    end = time.time()
    print('time = ', end - start)

    print('==================================================')
    
    start = time.time()
    weights = matrixSolution(X, y)
    print('weights = ', weights)
    end = time.time()
    print('time = ', end - start)
    
    
    
    

