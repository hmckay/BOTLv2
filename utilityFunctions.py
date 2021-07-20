import sys
import numpy as np
import pandas as pd
import time
import socket
import pickle
import time
import os.path
import math
from numpy.linalg import svd as SVD
from optparse import OptionParser
from scipy.spatial.distance import euclidean as euclid

# from Models import createModel,modelHistory
# from Models import modelMultiConceptTransfer as modelMulti
from Models import modelMultiConceptTransferHistoryRePro as modelMultiHistoryRepro
import preprocessData as preprocess
from datetime import datetime,timedelta
from sklearn import metrics
from scipy.stats import pearsonr

from scipy.linalg import norm
from sklearn.metrics.pairwise import polynomial_kernel as kernel
from Models.stsc_ulti import affinity_to_lap_to_eig, reformat_result, get_min_max
from Models.stsc_np import get_rotation_matrix as get_rotation_matrix_np

META_SIZE = 10
def calcError(y,preds):
    return metrics.r2_score(y,preds)

def modelReadyToSend(modelID,model,s,modelsSent):
    successFlag = 0
    if modelID not in modelsSent:
        successFlag = handshake(modelID,model,s,modelsSent)
    else:
        print("model already sent")
        return 1

    if successFlag:
        modelsSent.append(modelID)
        print("sucessfully sent model")
        return 1, modelsSent
    else:
        print("unsucessful send")
        return 0, modelsSent

def handshake(modelID,model,s,modelsSent):
    print(modelID)
    modelToSend = pickle.dumps(model)
    lenofModel = len(modelToSend)
    brokenBytes = [modelToSend[i:i+1024] for i in range(0,len(modelToSend),1024)]
    numPackets = len(brokenBytes)
    RTSmsg = 'RTS,'+str(modelID)+','+str(numPackets)+','+str(lenofModel)
    s.sendall(RTSmsg.encode())
    ack = s.recv(1024).decode()
    ackNumPackets = int(ack.split(',')[1])

    if ackNumPackets == numPackets:
        return sendModel(modelID,brokenBytes,s)
    return 0

def sendModel(modelID,brokenBytes,s):
    for i in brokenBytes:
        s.sendall(i)
        print("finished sending")
    recACK = s.recv(1024).decode()
    if modelID == int(recACK.split(',')[1]):
        return 1
    return 0

def readyToReceive(s,sourceModels,ID,METASTATS):
    s.sendall(('RTR,'+str(ID)).encode())
    data = s.recv(1024).decode()
    print("TARGET: num models to receive "+repr(data))
    ACKFlag = data.split(',')[0]
    numModels = int(data.split(',')[1])
    print("ready to recieve function called")
    if ACKFlag == 'ACK':
        if numModels == 0:
            s.sendall(('END').encode())
            return sourceModels,METASTATS
        s.sendall(('ACK').encode())
        for i in range(0,numModels):
            sourceModels,METASTATS = receiveModels(s,sourceModels,METASTATS)
    return sourceModels,METASTATS
# def readyToReceive(s,sourceModels,ID):
    # s.sendall(('RTR,'+str(ID)).encode())
    # data = s.recv(1024).decode()
    # print("TARGET: num models to receive "+repr(data))
    # ACKFlag = data.split(',')[0]
    # numModels = int(data.split(',')[1])
    # print("ready to recieve function called")
    # if ACKFlag == 'ACK':
        # if numModels == 0:
            # s.sendall(('END').encode())
            # return sourceModels
        # s.sendall(('ACK').encode())
        # for i in range(0,numModels):
            # sourceModels = receiveModels(s,sourceModels)
    # return sourceModels

def receiveModels(s,sourceModels,METASTATS):
    RTSInfo = s.recv(1024).decode()
    RTSFlag = RTSInfo.split(',')[0]
    sourceModID = RTSInfo.split(',')[1]
    numPackets = int(RTSInfo.split(',')[2])
    lenofModel = int(RTSInfo.split(',')[3])
    print("NUMBER OF PACKETS EXPECTING: "+str(numPackets))
    s.sendall(('ACK,'+str(numPackets)).encode())
    
    pickledModel = b''
    while (len(pickledModel) < lenofModel):
        pickledModel = pickledModel+s.recv(1024)
    s.sendall(('ACK,'+str(sourceModID)).encode())

    return storeSourceModel(sourceModID,pickledModel,sourceModels,METASTATS)
# def receiveModels(s,sourceModels):
    # RTSInfo = s.recv(1024).decode()
    # RTSFlag = RTSInfo.split(',')[0]
    # sourceModID = RTSInfo.split(',')[1]
    # numPackets = int(RTSInfo.split(',')[2])
    # lenofModel = int(RTSInfo.split(',')[3])
    # print("NUMBER OF PACKETS EXPECTING: "+str(numPackets))
    # s.sendall(('ACK,'+str(numPackets)).encode())
    
    # pickledModel = b''
    # while (len(pickledModel) < lenofModel):
        # pickledModel = pickledModel+s.recv(1024)
    # s.sendall(('ACK,'+str(sourceModID)).encode())

    # return storeSourceModel(sourceModID,pickledModel,sourceModels)

def storeSourceModel(sourceModID,pickledModel,sourceModels,METASTATS):
    print("picked model len is: "+str(len(pickledModel)))
    model = pickle.loads(pickledModel)
    sourceModels[sourceModID] = model
    METASTATS['modelsReceived'].append(sourceModID)
    print(model['model'])
    print("storing source model: "+str(sourceModID)) 
    return sourceModels,METASTATS
# def storeSourceModel(sourceModID,pickledModel,sourceModels):
    # print("picked model len is: "+str(len(pickledModel)))
    # model = pickle.loads(pickledModel)
    # sourceModels[sourceModID] = model
    # print(model['model'])
    # print("storing source model: "+str(sourceModID)) 
    # return sourceModels


def updatePADistanceMatrix(newID,otherModels,dM,distanceMetric):
    if dM is None: dM = dict()
    dM[newID]=dict()
    totalCalcs = 0
    for j in dM.keys():
        totalCalcs +=1
        distance = distanceMetric(otherModels[newID]['PCs'],otherModels[j]['PCs'])
        dM[newID][j] = distance
        dM[j][newID] = distance
    print("distance matrix is: ")
    print(dM)
    return dM,totalCalcs

def getSimMatrix(affDict,simKeys):
    simMatrix = np.zeros((len(simKeys),len(simKeys)))

    for idx,i in enumerate(simKeys):
        similarity = np.zeros(len(simKeys))
        for jdx,j in enumerate(simKeys):
            similarity[jdx]=affDict[i][j]
        simMatrix[idx]=similarity

    return simMatrix

def updateEuclidDistanceMatrix(newID,otherModels,dM,X,DROP_FIELDS,tLabel):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    dM[newID]=dict()

    for j in dM.keys():
        distance = euclid(otherModels[newID]['model'].predict(X),otherModels[j]['model'].predict(X))
        dM[newID][j] = distance
        dM[j][newID] = distance
    print("distance matrix is: ")
    print(dM)
    return dM

def self_tuning_spectral_clustering(affinity, modelKeys, get_rotation_matrix, min_n_cluster=None, max_n_cluster=None):
    
    w, v = affinity_to_lap_to_eig(affinity)
    min_n_cluster, max_n_cluster = get_min_max(w, min_n_cluster, max_n_cluster)
    if max_n_cluster > 10:
        max_n_cluster = 10
    re = []
    for c in range(min_n_cluster, max_n_cluster + 1):
        x = v[:, -c:]
        cost, r = get_rotation_matrix(x, c)
        re.append((cost, x.dot(r)))
        print('n_cluster: %d \t cost: %f' % (c, cost))
    COST, Z = sorted(re, key=lambda x: x[0])[0]
    return reformat_result(np.argmax(Z, axis=1), Z.shape[0],modelKeys)


def self_tuning_spectral_clustering_np(affinity, names, min_n_cluster=None, max_n_cluster=None):
    return self_tuning_spectral_clustering(affinity, names, get_rotation_matrix_np, min_n_cluster, max_n_cluster)

def getCluster(modelID,dM):
    modelKeys = list(dM.keys())
    affinityMatrix = getAffinityMatrix(dM)
    similarity_matrix = getSimMatrix(affinityMatrix,modelKeys)
    minClusters = None
    # groupedName, groupedID = self_tuning_spectral_clustering(similarity_matrix,similarity_matrix2, names, get_rotation_matrix_np, minClusters, None)
    print(similarity_matrix)
    groupedName, groupedID = self_tuning_spectral_clustering(similarity_matrix, modelKeys, get_rotation_matrix_np, minClusters, None)
    print(groupedName)
    print(groupedID)
    
    for c in groupedName:
        if modelID in c:
            return c
    else:
        return groupedName[0]


def getPerfCluster():
    pass

def checkToSend(modelID,window,existingModels,modelsSent,localDistanceMatrix,weightType,DROP_FIELDS,tLabel):
    totalCalcs = 0
    if len(modelsSent) < 2:
        localDistanceMatrix,tc = updatePADistanceMatrix(modelID,existingModels,localDistanceMatrix,principalAngles)
        return True,localDistanceMatrix,tc
    if 'PA' in weightType:
        localDistanceMatrix,tc = updatePADistanceMatrix(modelID,existingModels,localDistanceMatrix,principalAngles)
        totalCalcs +=tc
    else:
        localDistanceMatrix,tc = updateEuclidDistanceMatrix(modelID,existingModels,localDistanceMatrix,window,DROP_FIELDS,tLabel)
        totalCalcs +=tc
    clusterNewModel = getCluster(modelID,localDistanceMatrix)
    countModelsSent =len([v for v in clusterNewModel if v in modelsSent])
    print("models sent that are in the same cluster")
    print(clusterNewModel)
    print(countModelsSent)
    print(modelsSent)
    if countModelsSent > 1:
        return False,localDistanceMatrix,totalCalcs

    return True,localDistanceMatrix,totalCalcs

def addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,distanceMatrix=None):
    newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    uniqueModel = True
    totalCalcs = 0
    for newID in newSources:
        if WEIGHTTYPE == 'OLSFEPA':
            uniqueModel,sourceModels,distanceMatrix,tc = checkAngles(newID,sourceModels,distanceMatrix,PATHRESH)
            totalCalcs +=tc
        if uniqueModel:
            modelSet[newID] = sourceModels[newID]
            modelSet[newID]['delAWE']=0
    if 'PAC' in WEIGHTTYPE or 'CL' in WEIGHTTYPE:
        if lastModID is not None and lastModID not in modelSet.keys():
            if modelMultiHistoryRepro.isStable(lastModID,existingModels):
                modelSet[lastModID] = existingModels[lastModID]
                modelSet[lastModID]['delAWE']=0
    return modelSet,sourceModels,distanceMatrix,totalCalcs

def AWEaddToModelSet(df,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS):
    newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    for newID in newSources:
        numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
        modelSet[newID] = sourceModels[newID]
        # if numAWEModels<=META_SIZE-1:
        modelSet[newID]['delAWE'] = 0
        # else:
        while numAWEModels>META_SIZE-1:
            modelSet = modelMultiHistoryRepro.getBestAWEModels(df,modelSet,sourceModels[newID]['model'],newID,tLabel,DROP_FIELDS)
            numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
    if lastModID is not None:
        if modelMultiHistoryRepro.isStable(lastModID,existingModels):
            numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
            modelSet[lastModID] = existingModels[lastModID]
            # if numAWEModels<=META_SIZE-1:
            modelSet[lastModID]['delAWE'] = 0
            # else:
            while numAWEModels>META_SIZE-1:
                modelSet = modelMultiHistoryRepro.getBestAWEModels(df,modelSet,existingModels[lastModID]['model'],
                        lastModID,tLabel,DROP_FIELDS)
                numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
    return modelSet

def AddExpPaddToModelSet(df,idx,lastModID,existingModels,sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS):
    newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    metaModel = weights['metaModel']
    for newID in newSources:
        if not metaModel:
            metaModel = dict()
        # metaModel[newID]=1
        numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
        modelSet[newID] = sourceModels[newID]
        modelSet[newID]['delAddExp'] = 0
        while numAddExpModels>META_SIZE-1:
            # modelKeys = list(set(metaModel.keys())& set(modelSet.keys())
            minW = min(metaModel.values())
            minWeightID = [k for k,v in metaModel.items() if v == minW][0]
            minWeightID = min(metaModel.keys() & modelSet.keys(), key = metaModel.get)
            print("metamodel items:"+str(metaModel.items()))
            print("modelSet:"+str(modelSet))
            modelSet[minWeightID]['delAddExp']=1
            numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
            del metaModel[minWeightID]
        modelSet[newID]['delAddExp'] = 0
        weights['metaModel'] = metaModel
        metaModel = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,newID,modelSet[newID]['model'],weights)['metaModel']
    if lastModID is not None:
        if modelMultiHistoryRepro.isStable(lastModID,existingModels):
            numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
            modelSet[lastModID] = existingModels[lastModID]
            modelSet[lastModID]['delAddExp'] = 0
            while numAddExpModels>META_SIZE-1:
                minW = min(metaModel.values())
                minWeightID = [k for k,v in metaModel.items() if v == minW][0]
                minWeightID = min(metaModel.keys() & modelSet.keys(), key = metaModel.get)
                modelSet[minWeightID]['delAddExp']=1
                numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
                del metaModel[minWeightID]
            modelSet[lastModID]['delAddExp'] = 0
        else:
            del metaModel[lastModID]
    return modelSet,orderedModels,metaModel


def AddExpOaddToModelSet(df,idx,lastModID,existingModels,sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS):
    newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    metaModel = weights['metaModel']
    for newID in newSources:
        if not metaModel:
            metaModel = dict()
        # metaModel[newID]=1
        modelSet[newID] = sourceModels[newID]
        orderedModels.append(newID)
        while len(orderedModels)>META_SIZE-1:
            oldestID = orderedModels.pop(0)
            modelSet[oldestID]['delAddExp']=1
            del metaModel[oldestID]
        modelSet[newID]['delAddExp'] = 0
        weights['metaModel'] = metaModel
        metaModel = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,newID,modelSet[newID]['model'],weights)['metaModel']
    if lastModID is not None:
        if modelMultiHistoryRepro.isStable(lastModID,existingModels):
            modelSet[lastModID] = existingModels[lastModID]
            orderedModels.append(lastModID)
            if len(orderedModels)>META_SIZE-1:
                oldestID = orderedModels.pop(0)
                modelSet[oldestID]['delAddExp']=1
                del metaModel[oldestID]
            modelSet[lastModID]['delAddExp'] = 0
        else:
            del metaModel[lastModID]
    return modelSet,orderedModels,metaModel

def principalAngles(x,y):
    swapped = False
    if y.shape[1] < x.shape[1]:
        swapped=True
    yshape = y.shape[1]
    if y.shape[1] < x.shape[1]:
        yT = y.transpose()
        yTx = np.dot(yT,x)
        u,sig,v = SVD(yTx)
        angles = np.zeros(len(sig))
        yshape = x.shape[1]
    else:
        xT = x.transpose()
        xTy = np.dot(xT,y)
        u,sig,v = SVD(xTy)
        angles = np.zeros(len(sig))
    # print("sig:"+str(sig))
    # print("shape of x: "+str(x.shape[1]))
    # print("shape of y: "+str(y.shape[1]))
    # print("swapped: "+str(swapped))
    for idx,a in enumerate(sig):
        ang = a#np.round(a,decimals=14)
        print(idx,a)
        if a>=1:
            angles[idx] = np.arccos(1)
        else:
            angles[idx] = np.arccos(ang)
    tot = 0
    for a in angles:
        tot += np.cos(a)/yshape
    return (1 - tot)

def checkAngles(newID,sourceModels,distanceMatrix,PAThresh):
    totalCalcs = 0
    if distanceMatrix is None: distanceMatrix = dict()
    unique = True
    
    distanceMatrix,tc = updatePADistanceMatrix(newID,sourceModels,distanceMatrix,principalAngles)
    totalCalcs+=tc

    # distanceMatrix[newID]=dict()
    # for j in distanceMatrix.keys():
        # distance = principalAngles(sourceModels[newID]['PCs'],sourceModels[j]['PCs'])
        # distanceMatrix[newID][j] = distance
        # distanceMatrix[j][newID] = distance
    # print("distance matrix is: ")
    # print(distanceMatrix)
    
    affinityMatrix = getAffinityMatrix(distanceMatrix)
    print("affinity matrix is: ")
    print(affinityMatrix)

    if len(affinityMatrix) <=1:
        return unique,sourceModels,distanceMatrix,totalCalcs
    closestAffinity = sorted(affinityMatrix[newID].values(),reverse=True)[1]
    
    if closestAffinity >= PAThresh:
        unique = False

        print("Removing from model set"+str(newID))
        del distanceMatrix[newID]
        for j in distanceMatrix.keys():
            del distanceMatrix[j][newID]
        del sourceModels[newID]
    return unique,sourceModels,distanceMatrix,totalCalcs


def getAffinityMatrix(distanceMatrix):
    k=3
    affinityMatrix = dict()
    for i in distanceMatrix.keys():
        affinityMatrix[i]=dict()
        iZeroNeighbours = sum(1 for vals in distanceMatrix[i].values() if vals==0)
        if k > iZeroNeighbours and k <len(distanceMatrix.keys()):
            kNN = k
        elif k <=iZeroNeighbours and iZeroNeighbours < len(distanceMatrix.keys()):
            kNN = iZeroNeighbours
        else:
            kNN = len(distanceMatrix.keys())-1
        iNormaliser = sorted(distanceMatrix[i].values(),reverse=False)[kNN]

        for j in distanceMatrix.keys():
            jZeroNeighbours = sum(1 for vals in distanceMatrix[j].values() if vals==0)
            if k > jZeroNeighbours and k <len(distanceMatrix.keys()):
                kNN = k
            elif k <=jZeroNeighbours and jZeroNeighbours < len(distanceMatrix.keys()):
                kNN = jZeroNeighbours
            else:
                kNN = len(distanceMatrix.keys())-1
            jNormaliser = sorted(distanceMatrix[j].values(),reverse=False)[kNN]
            
            if iNormaliser == 0 and jNormaliser == 0:
                iNormaliser = 1
                jNormaliser = 1
            elif iNormaliser == 0:
                iNormaliser = jNormaliser
            elif jNormaliser == 0:
                jNormaliser = iNormaliser

            affinityMatrix[i][j] =  math.exp(-(distanceMatrix[i][j]**2)/(iNormaliser*jNormaliser))

    return affinityMatrix



# def AWEaddToModelSet(df,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS):
    # newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    # newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    # for newID in newSources:
        # numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
        # modelSet[newID] = sourceModels[newID]
        # if numAWEModels<META_SIZE-1:
            # modelSet[newID]['delAWE'] = 0
        # else:
            # modelSet = modelMultiHistoryRepro.getBestAWEModels(df,modelSet,sourceModels[newID]['model'],newID,tLabel,DROP_FIELDS)
    # if lastModID is not None and lastModID not in modelSet.keys():
        # if modelMultiHistoryRepro.isStable(lastModID,existingModels):
            # numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
            # modelSet[lastModID] = existingModels[lastModID]
            # if numAWEModels<META_SIZE-1:
                # modelSet[lastModID]['delAWE'] = 0
            # else:
                # modelSet = modelMultiHistoryRepro.getBestAWEModels(df,modelSet,existingModels[lastModID]['model'],
                        # lastModID,tLabel,DROP_FIELDS)
    # return modelSet

# def AddExpPaddToModelSet(df,idx,lastModID,existingModels,sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS):
    # newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    # newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    # metaModel = weights['metaModel']
    # print("modelset is")
    # print(modelSet)
    # for newID in newSources:
        # if not metaModel:
            # metaModel = dict()
        # # metaModel[newID]=1
        # modelSet[newID] = sourceModels[newID]
        # if len(metaModel.keys())>META_SIZE-1:
            # minW = min(metaModel.values())
            # minWeightID = [k for k,v in metaModel.items() if v == minW][0]
            # modelSet[minWeightID]['delAddExp']=1
            # del metaModel[minWeightID]
        # modelSet[newID]['delAddExp'] = 0
        # weights['metaModel'] = metaModel
        # metaModel = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,newID,modelSet[newID]['model'],weights)['metaModel']
        # print("heremeta")
        # print(metaModel)
    # if lastModID is not None:# and lastModID not in modelSet.keys():
        # if modelMultiHistoryRepro.isStable(lastModID,existingModels):
            # modelSet[lastModID] = existingModels[lastModID]
            # if len(metaModel.keys())>META_SIZE-1:
                # minW = min(metaModel.values())
                # minWeightID = [k for k,v in metaModel.items() if v == minW][0]
                # modelSet[minWeightID]['delAddExp']=1
                # del metaModel[minWeightID]
            # modelSet[lastModID]['delAddExp'] = 0
        # else:
            # del metaModel[lastModID]
    # return modelSet,orderedModels,metaModel


# def AddExpOaddToModelSet(df,idx,lastModID,existingModels,sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS):
    # newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    # newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    # metaModel = weights['metaModel']
    # print("modelset is")
    # print(modelSet)
    # for newID in newSources:
        # if not metaModel:
            # metaModel = dict()
        # # metaModel[newID]=1
        # modelSet[newID] = sourceModels[newID]
        # orderedModels.append(newID)
        # if len(orderedModels)>META_SIZE-1:
            # oldestID = orderedModels.pop(0)
            # modelSet[oldestID]['delAddExp']=1
            # del metaModel[oldestID]
        # modelSet[newID]['delAddExp'] = 0
        # weights['metaModel'] = metaModel
        # metaModel = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,newID,modelSet[newID]['model'],weights)['metaModel']
        # print("heremeta")
        # print(metaModel)
    # if lastModID is not None:# and lastModID not in modelSet.keys():
        # if modelMultiHistoryRepro.isStable(lastModID,existingModels):
            # modelSet[lastModID] = existingModels[lastModID]
            # orderedModels.append(lastModID)
            # if len(orderedModels)>META_SIZE-1:
                # oldestID = orderedModels.pop(0)
                # modelSet[oldestID]['delAddExp']=1
                # del metaModel[oldestID]
            # modelSet[lastModID]['delAddExp'] = 0
        # else:
            # del metaModel[lastModID]
    # return modelSet,orderedModels,metaModel


