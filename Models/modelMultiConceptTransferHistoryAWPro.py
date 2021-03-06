import numpy as np
from numpy.linalg import svd as SVD
import pandas as pd
import operator
from . import createModel
import datetime as dt
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.kernel_approximation import RBFSampler as RBF
from sklearn import metrics
from scipy.linalg import norm
from sklearn.metrics.pairwise import polynomial_kernel as kernel

PROB_THRESHOLD = 0
ACC_THRESHOLD = 0
STABLE_THRESHOLD = 0
SOURCE_MODELS = dict() 
def newHistory(acc,prob,stable,sourceModels,targetModel,startIDX,PCs):
    global ACC_THRESHOLD
    global PROB_THRESHOLD
    global STABLE_THRESHOLD
    global SOURCE_MODELS
    ACC_THRESHOLD = acc
    PROB_THRESHOLD = prob
    STABLE_THRESHOLD = stable
    SOURCE_MODELS = sourceModels
    modelInfo = {'model':targetModel,'usedFor':0,'PCs':PCs}
    existingModels = dict()
    existingModels[0] = modelInfo
    transitionMatrix=dict()
    transitionMatrix[0] = {}
    orderDetails = {'modelID':0,'startIDX':startIDX,'endIDX':None}
    modelOrder = []
    modelOrder.append(orderDetails)
    print("setting")
    print(ACC_THRESHOLD)
    return existingModels,transitionMatrix,0,modelOrder

def getLenUsedFor(modelID,existingModels):
    if modelID in existingModels:
        return existingModels[modelID]['usedFor']
    else:
        return 0

def isStable(modelID,existingModels):
    if existingModels[modelID]['usedFor']>=STABLE_THRESHOLD:
        return True
    return False

def getStableModels(existingModels):
    stable = dict((k,v) for k,v in existingModels.items() if v['usedFor']>=STABLE_THRESHOLD)
    #print stable
    return stable

def getBestAWEModels(df,modelSet,newBase,modelID,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    y = df[tLabel].copy()
    X = X.drop(tLabel,axis=1)
    mse = dict()
    print(modelSet)
    for k,v in modelSet.items():
        if modelSet[k]['delAWE'] == 0 and k != modelID:
            pred = modelSet[k]['model'].predict(X)
            mse[k] = metrics.mean_squared_error(y,pred)

    newMSE = metrics.mean_squared_error(y,newBase.predict(X))
    maxMSE = max(mse.values())
    if newMSE<maxMSE:
        res = [k for k,v in mse.items() if v==maxMSE][0]
        # print("model to delete: "+str(res))
        # print(len(sourceModels))
        # del modelSet[res]
        modelSet[res]['delAWE'] = 1
        # modelSet[modelID] = newBase
        modelSet[modelID]['delAWE'] = 0
    else:
        # modelSet[modelID] = newBase
        modelSet[modelID]['delAWE'] = 1
    
    return modelSet

def getCenteredK(df):
    # rbf_feature = RBF(gamma=1,random_state=1)
    # x_feat = rbf_feature.fit_transform(df)
    # kernelDF = pd.DataFrame(x_feat)
    # kernelMatrix = kernelDF.to_numpy()
    # print(df)
    kernelMatrix = df.to_numpy()
    # kernelT = kernelMatrix.transpose()
    # kTk = np.dot(kernelT,kernelMatrix)
    # kTk = kernel(kernelMatrix,kernelT, metric='rbf')
    kTk = kernel(kernelMatrix,kernelMatrix)
    # print(kTk.shape)
    N = kTk.shape[0]
    # print(N)
    centering = np.identity(N) - np.full(kTk.shape,(1/N))
    centeringT = centering.transpose()
    centeredKernel = np.dot(centering,np.dot(kTk,centeringT))

    return centeredKernel




def getPCs(df,DROP_FIELDS,varThresh):
    k = False
    normDF = df.copy()
    normDF = normDF.drop(DROP_FIELDS,axis=1)
    if k:
        xTx = getCenteredK(normDF)
        # rbf_feature = RBF(gamma=1,random_state=1)
        # x_feat = rbf_feature.fit_transform(normDF)
        # normDF = pd.DataFrame(x_feat)
    # print("normDF")
    # print(normDF)
    else:
        for col in normDF.columns:
            normDF[col] = normDF[col]-normDF[col].mean()
            std = normDF[col].std()
            if std != pd.np.nan and std != 0:
                normDF[col] = normDF[col]/normDF[col].std()
        x = normDF.to_numpy()
        xT = x.transpose()
        xTx = np.dot(xT,x)
    u,sig,v = SVD(xTx)

    sumDiag = 0
    totalSumDiag = np.sum(sig)
    numPCs = u.shape[0]
    for i in range(0,u.shape[0]):
        sumDiag += sig[i]
        varCap = 1-(sumDiag/totalSumDiag)
        # if varCap <= 0.001:
        if varCap <= varThresh:
            numPCs = i+1
            break
    if numPCs <= 1:
        numPCs = 3
    # print("rbf data:")
    # print(normDF)
    # print("principal components:")
    # print(numPCs)
    # print(u)
    # print("taking")
    # print(u[:,:numPCs])
    return u[:,:numPCs]


# def getPCs(df,DROP_FIELDS,k):
    # normDF = df.copy()
    # normDF = normDF.drop(DROP_FIELDS,axis=1)
    # if k:
        # rbf_feature = RBF(gamma=1,random_state=1)
        # x_feat = rbf_feature.fit_transform(normDF)
        # normDF = pd.DataFrame(x_feat)
    # # print("normDF")
    # # print(normDF)

    # for col in normDF.columns:
        # normDF[col] = normDF[col]-normDF[col].mean()
        # std = normDF[col].std()
        # if std != pd.np.nan and std != 0:
            # normDF[col] = normDF[col]/normDF[col].std()
    # x = normDF.to_numpy()
    # xT = x.transpose()
    # xTx = np.dot(xT,x)
    # u,sig,v = SVD(xTx)

    # sumDiag = 0
    # totalSumDiag = np.sum(sig)
    # numPCs = u.shape[0]
    # for i in range(0,u.shape[0]):
        # sumDiag += sig[i]
        # varCap = 1-(sumDiag/totalSumDiag)
        # if varCap <= 0.001:
            # numPCs = i+1
            # break
    # # print("rbf data:")
    # # print(normDF)
    # print("principal components:")
    # print(numPCs)
    # print(u)
    # print("taking")
    # print(u[:,:numPCs])
    # return u[:,:numPCs]

def addNewModel(existingModels,transitionMatrix,targetModel,prevModID,initTM,PCs):
    newID = len(existingModels)#+1
    modelInfo = {'model':targetModel,'usedFor':0,'PCs':PCs}
    existingModels[newID]= modelInfo
    transitionMatrix[newID]={}
    
    if not prevModID == 0:
        transitionMatrix[prevModID][newID] = 1
    
    return newID,existingModels,transitionMatrix

def addRepeatModel(transitionMatrix,newModID,prevModID):
    if newModID in transitionMatrix[prevModID]:
        transitionMatrix[prevModID][newModID] += 1
    else:
        transitionMatrix[prevModID][newModID] = 1
    return transitionMatrix

def searchModels(modelList,existingModels,df,tLabel,DROP_FIELDS):
    acc = 0
    nextModel = -1
    for m in modelList:
        if existingModels[m]['usedFor'] >= STABLE_THRESHOLD or acc == 0:
            testDF = df.copy()
            mod = existingModels[m]['model']
            res = createModel.initialPredict(testDF,mod,tLabel,DROP_FIELDS)
            thisAcc = metrics.r2_score(res[tLabel],res['predictions'])
            # if acc == 0:
                # acc=thisAcc
                # nextModel=m
        
            if thisAcc>acc:
                acc=thisAcc
                nextModel=m

    return acc, nextModel
def searchAllModels(modelList,existingModels,df,tLabel,DROP_FIELDS):
    acc = 0
    nextModel = -1
    print("STABLETHRESH: "+str(STABLE_THRESHOLD))
    print(existingModels)
    for m in modelList:
        if existingModels[m]['usedFor'] >= STABLE_THRESHOLD or acc == 0:
            testDF = df.copy()
            print(m)
            print(testDF)
            mod = existingModels[m]['model']
            res = createModel.initialPredict(testDF,mod,tLabel,DROP_FIELDS)
            thisAcc = metrics.r2_score(res[tLabel],res['predictions'])
            print(thisAcc)
            # if acc == 0:
                # acc=thisAcc
                # nextModel=m
            
            if thisAcc>acc:
                acc=thisAcc
                nextModel=m

    return acc, nextModel


def compareModels(data,m1,m2,tLabel,DROP_FIELDS):
    d1 = data.copy()
    d2 = data.copy()
    y1 = createModel.initialPredict(d1,m1,tLabel,DROP_FIELDS)['predictions']
    y2 = createModel.initialPredict(d2,m2,tLabel,DROP_FIELDS)['predictions']
   
    diffs = abs(y1-y2)
    maxDiff = max(diffs)
    return maxDiff

def updateLastModelUsage(modelOrder,existingModels,endIDX):
    modelOrder[-1]['endIDX'] = endIDX
    lastModel = modelOrder[-1]
    indexDiff = 0
    if lastModel['endIDX'] != lastModel['startIDX']:
        indexDiff = (lastModel['endIDX']-lastModel['startIDX'])
    existingModels[lastModel['modelID']]['usedFor'] += indexDiff
    return modelOrder, existingModels


def updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX,endLen):
    modelOrder[-1]['endIDX'] = startIDX-endLen
    lastModel = modelOrder[-1]
    indexDiff = 0
    if lastModel['endIDX'] != lastModel['startIDX']:
        indexDiff = (lastModel['endIDX']-lastModel['startIDX'])
    existingModels[lastModel['modelID']]['usedFor'] += indexDiff

    if indexDiff < STABLE_THRESHOLD and len(modelOrder)>1:
        falseFrom = modelOrder[-2]['modelID']
        falseTo = modelOrder[-1]['modelID']
        if falseTo in transitionMatrix[falseFrom]:
            if transitionMatrix[falseFrom][falseTo]>0:
                transitionMatrix[falseFrom][falseTo] -= 1
    back = -1
    while indexDiff < STABLE_THRESHOLD:
        if len(modelOrder) <= (back*-1):
            lastSuccess = modelOrder[back]
            currentModID = lastSuccess['modelID']
            break
        else:
            lastSuccess = modelOrder[back-1]
        currentModID = lastSuccess['modelID']
        indexDiff = (lastSuccess['endIDX']-lastSuccess['startIDX'])
        back -= 1

    return modelOrder,existingModels,transitionMatrix, currentModID

def reuseModels(existingModels,DF,tLabel,DROP_FIELDS):
    df = DF.copy()
    acc,nextModel = searchAllModels(existingModels,existingModels,df,tLabel,DROP_FIELDS)
    print("reuse model:"+str(nextModel)+","+str(acc))
    if acc<ACC_THRESHOLD:
        # print("CANT reuse model "+str(nextModel)+": "+str(acc))
        return False
    else:
        # print("can reuse model "+str(nextModel)+": "+str(acc))
        return True


def tempModel(existingModels,transitionMatrix,modelOrder,DF,currentModID,tLabel,DROP_FIELDS,startIDX,initTM,tempModel,DEFAULT_PRED):
    if not tempModel:
        # modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX)
        modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX,len(DF))
    df = DF.copy()
    m = createModel.createPipeline(df,tLabel,DROP_FIELDS,DEFAULT_PRED)
    nextModel = currentModID
    
    return nextModel,existingModels,transitionMatrix,modelOrder,m,True


# def nextModels(existingModels,transitionMatrix,modelOrder,DF,currentModID,tLabel,DROP_FIELDS,startIDX,initTM,weightType,tempModel,DEFAULT_PRED):
def nextModels(existingModels,transitionMatrix,modelOrder,DF,currentModID,tLabel,DROP_FIELDS,startIDX,initTM,
        weightType,tempModel,varThresh,DEFAULT_PRED):
    newTarget=False
    if not tempModel:
        # modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX)
        modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX,len(DF))
    df = DF.copy()
    successors = transitionMatrix.get(currentModID).copy()
    total = sum(successors.values())
    if total == 0:
        # acc,nextModel = searchModels(existingModels,existingModels,df,tLabel,DROP_FIELDS)
        acc,nextModel = searchAllModels(existingModels,existingModels,df,tLabel,DROP_FIELDS)
        # print("model "+str(nextModel)+": "+str(acc))
        if acc<ACC_THRESHOLD:
            repeatModel = 0
            # print("generating new model becuase no successors")
            # m = createModel.createPipeline(df,target,DROP_FIELDS)
            # nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,m,currentModID,initTM)
            targetModel = createModel.createPipeline(df,tLabel,DROP_FIELDS,DEFAULT_PRED)
            newTarget =True
            if 'PA' in weightType:
                PCs = getPCs(df,DROP_FIELDS,varThresh)
            else:
                PCs = None
            nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,
                    targetModel,currentModID,initTM,PCs)
        else:
            # print("historical-reactive")
            transitionMatrix = addRepeatModel(transitionMatrix,nextModel,currentModID)
            newTarget = False
        #print transitionMatrix

        orderDetails = {'modelID':nextModel,'startIDX':startIDX,'endIDX':None}
        modelOrder.append(orderDetails)
        return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['model'],True

    successors.update((x,y/total) for x,y in list(successors.items()))
    max_val = max(iter(successors.items()),key=operator.itemgetter(1))[1]
    nextModels = []
    nextModels = [k for k, v in successors.items() if v == max_val and max_val >= PROB_THRESHOLD]
    #print "next models: " + str(nextModels)
    nextModel = -1
    acc = 0
    repeatModel = 1
    print("acc thresh:")
    print(ACC_THRESHOLD)
    if len(nextModels)>0:
        # if 0 not in nextModels:# and isStable(0,existingModels):
            # nextModels.append(0)
        # nextModels.append(1)
        acc,nextModel = searchModels(nextModels,existingModels,df,tLabel,DROP_FIELDS)
        # print("proactive acc: "+str(acc))
    # else:
    if acc < ACC_THRESHOLD:
        # print("historical-reactive")
        # acc, nextModel = searchModels(existingModels,existingModels,df,tLabel,DROP_FIELDS)
        acc, nextModel = searchAllModels(existingModels,existingModels,df,tLabel,DROP_FIELDS)
    
    if acc < ACC_THRESHOLD:
        #learn new concept
        #print "learn new model acc: "+str(acc)
        repeatModel = 0
        # print("generating new model")
        # m = createModel.createPipeline(df,target,DROP_FIELDS)
        # nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,m,currentModID,initTM)
        targetModel = createModel.createPipeline(df,tLabel,DROP_FIELDS,DEFAULT_PRED)
        if 'PA' in weightType:
            PCs = getPCs(df,DROP_FIELDS,varThresh)
        else:
            PCs = None
        nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,
                targetModel,currentModID,initTM,PCs)
    print(str(nextModel)+": "+str(acc))

    if repeatModel == 1:
        transitionMatrix = addRepeatModel(transitionMatrix,nextModel,currentModID)
    
    orderDetails = {'modelID':nextModel,'startIDX':startIDX,'endIDX':None}
    modelOrder.append(orderDetails)

    return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['model'],True

# def nextModels(existingModels,transitionMatrix,modelOrder,DF,currentModID,tLabel,DROP_FIELDS,startIDX,initTM,weightType):
    # modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX)
    # df = DF.copy()
    # targetModel = createModel.createPipeline(df,tLabel,DROP_FIELDS)
    # if 'OLSKPAC' in weightType:
        # PCs = getPCs(df,DROP_FIELDS,True)
    # elif 'OLSPAC' in weightType:
        # PCs = getPCs(df,DROP_FIELDS,False)
    # else:
        # PCs = None
    # nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,
            # targetModel,currentModID,initTM,PCs)
    # orderDetails = {'modelID':nextModel,'startIDX':startIDX,'endIDX':None}
    # modelOrder.append(orderDetails)

    # return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['targetModel']



