
import sys
import numpy as np
import pandas as pd
import time
import socket
import pickle
import time
import os.path

from Models import createModel,modelHistory
from Models import modelMultiConceptTransfer as modelMulti
from Models import modelMultiConceptTransferHistoryADWIN as modelMultiHistory
import preprocessData as preprocess
from datetime import datetime,timedelta
from sklearn import metrics
from scipy.stats import pearsonr
from pyadwin import Adwin
from utilityFunctions import *



DROP_FIELDS = []# ['Time','predictions']
tLabel=''#'FollowingTime'

INIT_DAYS = 0
STABLE_SIZE = 0
MAX_WINDOW = 0
ADWIN_DELTA = 0
PORT = 0
SIM_FROM = 0
SIM_TO = 0
RUNID=0
NUMSTREAMS = 0
WEIGHTTYPE = ''
CULLTHRESH = 0
MITHRESH = 0
FP = ''
modelsSent = []
sentFlag = 0
UID = ''
DEFAULT_PRED = ''
EPSILON = 0
META_SIZE = 10
PATHRESH = 0
PCAVAR = 0
METASTATS = dict()
MODELUSAGE=dict()


# def calcError(y,preds):
    # return metrics.r2_score(y,preds)

# def modelReadyToSend(modelID,model,s):
    # successFlag = 0
    # if modelID not in modelsSent:
        # successFlag = handshake(modelID,model,s)
    # else:
        # print("model already sent")
        # return 1

    # if successFlag:
        # modelsSent.append(modelID)
        # print("sucessfully sent model")
        # return 1
    # else:
        # print("unsucessful send")
        # return 0

# def handshake(modelID,model,s):
    # print(modelID)
    # modelToSend = pickle.dumps(model)
    # lenofModel = len(modelToSend)
    # brokenBytes = [modelToSend[i:i+1024] for i in range(0,len(modelToSend),1024)]
    # numPackets = len(brokenBytes)
    # RTSmsg = 'RTS,'+str(modelID)+','+str(numPackets)+','+str(lenofModel)
    # s.sendall(RTSmsg.encode())
    # ack = s.recv(1024).decode()
    # ackNumPackets = int(ack.split(',')[1])

    # if ackNumPackets == numPackets:
        # return sendModel(modelID,brokenBytes,s)
    # return 0

# def sendModel(modelID,brokenBytes,s):
    # for i in brokenBytes:
        # s.sendall(i)
        # print("finished sending")
    # recACK = s.recv(1024).decode()
    # if modelID == int(recACK.split(',')[1]):
        # return 1
    # return 0

# def readyToReceive(s,sourceModels):
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

# def storeSourceModel(sourceModID,pickledModel,sourceModels):
    # print("picked model len is: "+str(len(pickledModel)))
    # model = pickle.loads(pickledModel)
    # sourceModels[sourceModID] = model
    # print(model['model'])
    # return sourceModels
    
# def addToModelSet(lastModID,existingModels,sourceModels,modelSet):
    # newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    # newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    # for newID in newSources:
        # modelSet[newID] = sourceModels[newID]
        # modelSet[newID]['delAWE']=0
        # modelSet[newID]['prune']=False
    # if lastModID is not None:
        # if modelMultiHistory.isStable(lastModID,existingModels):
            # modelSet[lastModID] = existingModels[lastModID]
            # modelSet[lastModID]['delAWE']=0
            # modelSet[lastModID]['prune']=False
    # return modelSet

# def AWEaddToModelSet(df,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS):
    # newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    # newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    # for newID in newSources:
        # numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
        # modelSet[newID] = sourceModels[newID]
        # # if numAWEModels<=META_SIZE-1:
        # modelSet[newID]['delAWE'] = 0
        # # else:
        # while numAWEModels>META_SIZE-1:
            # modelSet = modelMultiHistory.getBestAWEModels(df,modelSet,sourceModels[newID]['model'],newID,tLabel,DROP_FIELDS)
            # numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
    # if lastModID is not None:
        # if modelMultiHistory.isStable(lastModID,existingModels):
            # numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
            # modelSet[lastModID] = existingModels[lastModID]
            # # if numAWEModels<=META_SIZE-1:
            # modelSet[lastModID]['delAWE'] = 0
            # # else:
            # while numAWEModels>META_SIZE-1:
                # modelSet = modelMultiHistory.getBestAWEModels(df,modelSet,existingModels[lastModID]['model'],
                        # lastModID,tLabel,DROP_FIELDS)
                # numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
    # return modelSet

# def AddExpPaddToModelSet(df,idx,lastModID,existingModels,sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS):
    # newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    # newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    # metaModel = weights['metaModel']
    # for newID in newSources:
        # if not metaModel:
            # metaModel = dict()
        # # metaModel[newID]=1
        # numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
        # modelSet[newID] = sourceModels[newID]
        # modelSet[newID]['delAddExp'] = 0
        # while numAddExpModels>META_SIZE-1:
            # # modelKeys = list(set(metaModel.keys())& set(modelSet.keys())
            # minW = min(metaModel.values())
            # minWeightID = [k for k,v in metaModel.items() if v == minW][0]
            # minWeightID = min(metaModel.keys() & modelSet.keys(), key = metaModel.get)
            # print("metamodel items:"+str(metaModel.items()))
            # print("modelSet:"+str(modelSet))
            # modelSet[minWeightID]['delAddExp']=1
            # numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
            # del metaModel[minWeightID]
        # modelSet[newID]['delAddExp'] = 0
        # weights['metaModel'] = metaModel
        # metaModel = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,newID,modelSet[newID]['model'],weights)['metaModel']
    # if lastModID is not None:
        # if modelMultiHistory.isStable(lastModID,existingModels):
            # numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
            # modelSet[lastModID] = existingModels[lastModID]
            # modelSet[lastModID]['delAddExp'] = 0
            # while numAddExpModels>META_SIZE-1:
                # minW = min(metaModel.values())
                # minWeightID = [k for k,v in metaModel.items() if v == minW][0]
                # minWeightID = min(metaModel.keys() & modelSet.keys(), key = metaModel.get)
                # modelSet[minWeightID]['delAddExp']=1
                # numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
                # del metaModel[minWeightID]
            # modelSet[lastModID]['delAddExp'] = 0
        # else:
            # del metaModel[lastModID]
    # return modelSet,orderedModels,metaModel


# def AddExpOaddToModelSet(df,idx,lastModID,existingModels,sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS):
    # newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    # newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    # metaModel = weights['metaModel']
    # for newID in newSources:
        # if not metaModel:
            # metaModel = dict()
        # # metaModel[newID]=1
        # modelSet[newID] = sourceModels[newID]
        # orderedModels.append(newID)
        # while len(orderedModels)>META_SIZE-1:
            # oldestID = orderedModels.pop(0)
            # modelSet[oldestID]['delAddExp']=1
            # del metaModel[oldestID]
        # modelSet[newID]['delAddExp'] = 0
        # weights['metaModel'] = metaModel
        # metaModel = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,newID,modelSet[newID]['model'],weights)['metaModel']
    # if lastModID is not None:
        # if modelMultiHistory.isStable(lastModID,existingModels):
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

# # def updateAWEModelSet(df,newBase,modelID,modelSet,META_SIZE,tLabel,DROP_FIELDS):
    # # numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 1])
    # # if numAWEModels<META_SIZE:
        # # modelSet[modelID]['delAWE'] = 0
    # # else:
        # # modelSet = modelMultiHistory.getBestAWEModels(df,modelSet,newBase,modelID,tLabel,DROP_FIELDS)
    # # return modelSet


def runADWIN(df,sourceModels,tLabel,weightType,s):
    global modelsSent
    global METASTATS
    global MODELUSAGE
    adwin=Adwin(MAX_WINDOW,ADWIN_DELTA)
    buildTMod = False
    sourceAlpha = 1
    targetModel = None
    existingModels = dict()
    transitionMatrix = dict()
    multiWeights = []
    weight = dict()
    weights = dict()
    for m in sourceModels:
        weight[m] = 1.0/len(sourceModels)
    print("initial weights:")
    print(weight)
    weights['sourceR2s']=weight
    weights['totalR2']=len(sourceModels)
    weights['metaModel'] = dict()
    multiWeights.append(weights)
    numModelsUsed = []
    week = 0
    drifts = dict()
    storeFlag = False
    modelID = 0
    lastModID = 0
    modelCount = 0
    modelOrder = []
    windowSize = 0
    conceptSize = 0
    ofUse = 0
    sentFlag = 0
    window = pd.DataFrame(columns = df.columns)
    window = window.astype(df.dtypes.to_dict())
    historyData = pd.DataFrame(columns = df.columns)
    historyData = historyData.astype(df.dtypes.to_dict())
    startIDX = df.index.min()+INIT_DAYS+1
    # startDoW = df.index.min()%20
    startDoW = df.index.min()%INIT_DAYS
    retrainFlag=False
    updateFlag=False
    distanceMatrix = None
    localDistanceMatrix = None
    affinityMatrix = None
    existingBases = None
    newTarget = False
    groupedNames = None
    modelSet = dict()
    orderedModels = []
    notSent = []
    numBases=0
    currentBases=[]
    METASTATS['currentBaseLearnerIDs'] = []

    modelStart = []
    modelStart.append(startIDX)
    p = pd.DataFrame(columns = df.columns)
    if 'AWE' in weightType:
        for k in sourceModels.keys():
            modelSet[k] = sourceModels[k]
            modelSet[k]['delAWE'] = 0
        # modelSet = AWEaddToModelSet(df,None,None,sourceModels,modelSet,tLabel,DROP_FIELDS)
    elif 'AddExp' in weightType:
        print("here initinti")
        # modelSet,orderedModels,weights['metaModel'] = AddExpOaddToModelSet(df,0,None,None,sourceModels,modelSet,
                # orderedModels,weights,tLabel,DROP_FIELDS)
    else:
        # modelSet = addToModelSet(None,None,sourceModels,modelSet)
        modelSet,sourceModels,distanceMatrix,tc = addToModelSet(None,None,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,distanceMatrix)
        METASTATS['totalPACalcs']+=tc
        if tc > 0:
            METASTATS['triggerPACalcs']+=1
    
    for idx in df.index:
        print(idx,modelID)
        METASTATS['totalNumInstances']+=1
        if modelID not in MODELUSAGE.keys():
            MODELUSAGE[modelID]=0
        if MODELUSAGE[modelID]>=STABLE_SIZE and modelID not in METASTATS['stableLocal']:
            METASTATS['stableLocal'].append(modelID)
        conceptSize+=1
        if idx%INIT_DAYS == startDoW and storeFlag == False:
            storeFlag = True
            week += 1
        
        # if ofUse >= STABLE_SIZE and not sentFlag:
        # if (modelMultiHistory.getLenUsedFor(modelID,existingModels)) and (modelID not in modelsSent):
        # if ((modelMultiHistory.getLenUsedFor(modelID,existingModels)+ofUse)>=STABLE_SIZE) and (modelID not in modelsSent):
        if (ofUse)>=STABLE_SIZE and (modelID not in modelsSent) and (modelID not in notSent):
            print("trying to send "+str(modelID))
            print(idx)
            # if 'OLSKPAC' in weightType:
                # PCs = modelMultiHistory.getPCs(historyData[(-1*MAX_WINDOW):],DROP_FIELDS,True)
            # elif 'OLSPAC' in weightType:
                # PCs = modelMultiHistory.getPCs(historyData[(-1*MAX_WINDOW):],DROP_FIELDS,False)
            # else:
                # PCs = None
            # existingModels[modelID]['PCs'] = PCs
            # modelInfo = {'model':targetModel,'PCs':PCs}
            # sentFlag = modelReadyToSend(modelID,modelInfo,s)
            modelInfoToSend = {'model':targetModel,'PCs':existingModels[modelID]['PCs']}
            # sentFlag,modelsSent = modelReadyToSend(modelID,modelInfoToSend,s,modelsSent)
            checkSendFlag = True
            if 'Red' in weightType:
                checkSendFlag, localDistanceMatrix,tc = checkToSend(modelID,window,existingModels,modelsSent,localDistanceMatrix,
                        weightType,DROP_FIELDS,tLabel)
                METASTATS['totalPACalcs']+=tc
                if tc > 0:
                    METASTATS['triggerPACalcs']+=1
            if not checkSendFlag:
                notSent.append(modelID)
            else:
                sentFlag,modelsSent = modelReadyToSend(modelID,modelInfoToSend,s,modelsSent)
            if sentFlag and modelID not in METASTATS['modelsSent']:
                METASTATS['modelsSent'].append(modelID)
        
        #first window of data
        if len(historyData) < MAX_WINDOW:
            if (len(historyData)%(int(MAX_WINDOW/4)) == (int(MAX_WINDOW/4)-1)) and len(sourceModels)>0:
                weight = modelMulti.updateInitialWeights(historyData,sourceModels,tLabel,
                        DROP_FIELDS,weightType,CULLTHRESH,MITHRESH)
                if 'PAC' in weightType or 'CL2' in weightType:
                    METASTATS['totalPACalcs']+=weight[-1][1]
                    if weight[-1][1] > 0:
                        METASTATS['triggerPACalcs']+=1
                    METASTATS['totalOtherMetricCalcs']+=weight[-1][0]
                    if weight[-1][0]>0:
                        METASTATS['triggerMetricCalcs']+=1
                else:
                    METASTATS['totalOtherMetricCalcs']+=weight[-1]
                    if weight[-1]>0:
                        METASTATS['triggerMetricCalcs']+=1
                METASTATS['metaWeightUpdate']+=1
            if not sourceModels:
                prediction,numBases,currentBases = modelMulti.defaultInstancePredict(df,idx,DEFAULT_PRED)
                METASTATS['numberBaseLearners'].append(numBases)
                METASTATS['baseLearnerList'].append(currentBases)
                if not set(METASTATS['currentBaseLearnerIDs']) == set(currentBases):
                    METASTATS['baseLearnerChange']+=1
                    METASTATS['currentBaseLearnerIDs']=currentBases
                targetPred = prediction
            else:
                #"making prediction, source Models: "+str(sourceModels)
                if ('PAC' in weightType or 'CL' in weightType) and len(sourceModels)>=1 and len(historyData)>=20:
                    prediction,distanceMatrix,affinityMatrix,groupedNames,numBases,currentBases,tc = modelMulti.initialInstancePredict(df,idx,
                            sourceModels,weight,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,distanceMatrix,
                            affinityMatrix,groupedNames)
                    METASTATS['numberBaseLearners'].append(numBases)
                    METASTATS['baseLearnerList'].append(currentBases)
                    if not set(METASTATS['currentBaseLearnerIDs']) == set(currentBases):
                        METASTATS['baseLearnerChange']+=1
                        METASTATS['currentBaseLearnerIDs']=currentBases
                else:
                    print("in this else because not")
                    prediction,numBases,currentBases,tc = modelMulti.initialInstancePredict(df,idx,sourceModels,weight,tLabel,
                            DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,distanceMatrix,affinityMatrix,groupedNames)
                    METASTATS['numberBaseLearners'].append(numBases)
                    METASTATS['baseLearnerList'].append(currentBases)
                    if not set(METASTATS['currentBaseLearnerIDs']) == set(currentBases):
                        METASTATS['baseLearnerChange']+=1
                        METASTATS['currentBaseLearnerIDs']=currentBases

                # prediction = modelMulti.initialInstancePredict(df,idx,modelSet,weight,tLabel,
                        # DROP_FIELDS,weightType,CULLTHRESH,MITHRESH)
                targetPred,numBases,currentBases = modelMulti.defaultInstancePredict(df,idx,DEFAULT_PRED)
            historyData = historyData.append(prediction)
            p = p.append(targetPred)
            # if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
            if 'OLS' in weightType:
                if not sourceModels:
                    numModels = 0
                else:
                    modelWeights = weights['sourceR2s']
                    numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                    numModelsUsed.append(numModels)


            #    # resultFile = open('performanceVsCost3'+str(ID)+str(weightType)+'.csv','a')
            #    # wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
            #    # if not sourceModels:
            #        # numModels = 0
            #    # else:
            #        # modelWeights = weights['sourceR2s']
            #        # numModels = sum(1 for val in modelWeights.values() if val != 0)
            #    # resultFile.write(str(idx)+','+str(wErr)+','+str(numModels)+'\n')
            #    # resultFile.close()
        #one window of data received
        elif (len(historyData) == MAX_WINDOW) and not buildTMod:
            # if (weightType != 'OLS' and weightType != 'OLSFE' and weightType != 'Ridge' and 
                    # weightType != 'NNLS' and weightType != 'OLSFEMI' and weightType != 'OLSCL'):
            if ('OLS' not in weightType and weightType != 'Ridge' and weightType != 'NNLS'):
                weights['sourceR2s']=weight
                weights['totalR2']=sum(weight.values())
            else: 
                weights = weight
            multiWeights.append(weights)
            #print "building first target model"
            buildTMod = True
            print("building target model: "+str(idx))
            #receive models from controller
            print("RtR: "+str(idx))
            # sourceModels = readyToReceive(s,sourceModels,ID)
            sourceModels,METASTATS = readyToReceive(s,sourceModels,ID,METASTATS)
            targetModel = createModel.createPipeline(historyData,tLabel,DROP_FIELDS,DEFAULT_PRED)
            if 'AWE' in weightType:
                modelSet = AWEaddToModelSet(historyData,None,None,sourceModels,modelSet,tLabel,DROP_FIELDS)
            elif 'AddExpO' in weightType:
                # if 'metaModel' not in weights.keys():
                weights['metaModel']=dict()
                modelSet,orderedModels,weights['metaModel'] = AddExpOaddToModelSet(historyData,idx,None,None,sourceModels,modelSet,
                        orderedModels,weights,tLabel,DROP_FIELDS)
                METASTATS['metaWeightUpdate']+=1
                # weights['metaModel'][modelID]=1
                # weights = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,modelID,weights)
            elif 'AddExpP' in weightType:
                # if 'metaModel' not in weights.keys():
                weights['metaModel']=dict()
                modelSet,orderedModels,weights['metaModel'] = AddExpPaddToModelSet(historyData,idx,None,None,sourceModels,modelSet,
                        orderedModels,weights,tLabel,DROP_FIELDS)
                METASTATS['metaWeightUpdate']+=1

            else:
                # modelSet = addToModelSet(None,None,sourceModels,modelSet)
                modelSet,sourceModels,distanceMatrix,tc = addToModelSet(None,None,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,distanceMatrix)
            # if 'OLSKPAC' in weightType:
                # PCs = modelMultiHistory.getPCs(historyData,DROP_FIELDS,PCAVAR)
            # elif 'OLSPAC' in weightType:
            if 'PA' in weightType:
                PCs = modelMultiHistory.getPCs(historyData,DROP_FIELDS,PCAVAR)
            else:
                PCs = None

            # weights,distanceMatrix,affinityMatrix,groupedNames = modelMulti.calcWeights(historyData,sourceModels,
                    # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,None,None,PCs,
                    # None,None,newTarget,groupedNames)
            if 'AddExp' not in weightType:
                weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(historyData,modelSet,
                        targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,None,modelID,PCs,
                        distanceMatrix,affinityMatrix,newTarget,groupedNames,METASTATS['currentBaseLearnerIDs'])
                if 'PAC' in weightType or 'CL2' in weightType:
                    METASTATS['totalPACalcs']+=tc[1]
                    if tc[1] > 0:
                        METASTATS['triggerPACalcs']+=1
                    METASTATS['totalOtherMetricCalcs']+=tc[0]
                    if tc[0]>0:
                        METASTATS['triggerMetricCalcs']+=1
                else:
                    METASTATS['totalOtherMetricCalcs']+=tc
                    if tc>0:
                        METASTATS['triggerMetricCalcs']+=1
                METASTATS['metaWeightUpdate']+=1
            # if 'AddExp' in weightType:
                # weights['metaModel'] = dict()
                # for mID in modelSet.keys():
                    # weights['metaModel'][mID]=1
                # if modelID not in weights['metaModel'].keys():
                    # weights = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,modelID,weights)
                # else:
                    # weights = modelMulti.calcUpdateAddExpWeight(df,idx,modelSet,weights['metaModel'],targetModel,
                            # modelID,tLabel,DROP_FIELDS)
            if 'AWE' in weightType:
                modelSet = weights['AWEmodelSet']
            newTarget = False
            # if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
            if 'OLS' in weightType and 'AddExp' not in weightType:
                modelWeights = weights['sourceR2s']
                numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                numModelsUsed.append(numModels)
            # if weightType == 'OLSFE' or weightType =='OLS':
            #    # resultFile = open('performanceVsCost3'+str(ID)+str(weightType)+'.csv','a')
            #    # wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
            #    # modelWeights = weights['sourceR2s']
            #    # numModels = sum(1 for val in modelWeights.values() if val != 0)
            #    # resultFile.write(str(idx)+','+str(wErr)+','+str(numModels+1)+'\n')
            #    # resultFile.close()
            print("first prediction with target model: "+str(idx))
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            prediction,numBases,currentBases = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType,None,modelID,PCs)
            METASTATS['numberBaseLearners'].append(numBases)
            METASTATS['baseLearnerList'].append(currentBases)
            if not set(METASTATS['currentBaseLearnerIDs']) == set(currentBases):
                METASTATS['baseLearnerChange']+=1
                METASTATS['currentBaseLearnerIDs']=currentBases
            
            historyData = historyData.append(prediction)
            p = p.append(targetPred)
            
            ofUse += 1
            #calculate ensemble weights - return dict

            print("first alpha value is: " +str(weights))

            ##########CREATE NEW HISTORY FOR ENSEMBLE WEIGHTS
            existingModels,transitionMatrix,modelID,modelOrder = modelMultiHistory.newHistory(STABLE_SIZE,sourceModels,
                    targetModel,startIDX,PCs)
            if modelID not in MODELUSAGE.keys():
                MODELUSAGE[modelID]=1
            else:
                MODELUSAGE[modelID]+=1
            drifts[modelCount] = week
            modelCount = 1
            print(modelID)
        #continue with ADWIN
        else:
            print("in else"+str(modelID))
            if weightType == 'OLSCL2' or 'OLSPAC' in weightType or 'OLSKPAC' in weightType:
                stableLocals = modelMultiHistory.getStableModels(existingModels)
            else:
                stableLocals = None
            # if 'OLSPAC' in weightType or 'OLSKPAC' in weightType:
            if 'PA' in weightType:
                PCs = existingModels[modelID]['PCs']
            else:
                PCs = None
            # if idx%25 == 0 and (weightType == 'OLS' or weightType == 'OLSFE' or 
                    # weightType == 'OLSFEMI' or weightType == 'Ridge' or weightType == 'OLSCL'):
            if 'AddExp' in weightType:
                tc=0
                if modelID not in weights['metaModel'].keys():
                    weights,tc = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,modelID,targetModel,weights)
                else:
                    weights,tc = modelMulti.calcUpdateAddExpWeight(df,idx,modelSet,weights['metaModel'],targetModel,
                            modelID,tLabel,DROP_FIELDS)
                METASTATS['totalOtherMetricCalcs']+=tc
                if tc>0:
                    METASTATS['triggerMetricCalcs']+=1
                METASTATS['metaWeightUpdate']+=1

            if idx%(int(MAX_WINDOW/2)) == 0 and ('OLS' in weightType or weightType == 'Ridge'):
                # stableLocals = modelMultiHistory.getStableModels(existingModels)
                # weights = modelMulti.calcWeights(historyData[(-1*STABLE_SIZE):],sourceModels,
                        # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,PCs)
                # weights,distanceMatrix,affinityMatrix,groupedNames = modelMulti.calcWeights(historyData[(-1*STABLE_SIZE):],
                        # sourceModels,targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,PCs,
                        # distanceMatrix,affinityMatrix,newTarget,groupedNames)
                if 'AddExp' not in weightType:
                    weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(
                            historyData[(-1*STABLE_SIZE):],modelSet,targetModel,tLabel,DROP_FIELDS,weightType,
                            CULLTHRESH,MITHRESH,stableLocals,modelID,PCs,
                            distanceMatrix,affinityMatrix,newTarget,groupedNames,METASTATS['currentBaseLearnerIDs'])
                    if 'PAC' in weightType or 'CL2' in weightType:
                        METASTATS['totalPACalcs']+=tc[1]
                        if tc[1] > 0:
                            METASTATS['triggerPACalcs']+=1
                        METASTATS['totalOtherMetricCalcs']+=tc[0]
                        if tc[0]>0:
                            METASTATS['triggerMetricCalcs']+=1
                    else:
                        METASTATS['totalOtherMetricCalcs']+=tc
                        if tc>0:
                            METASTATS['triggerMetricCalcs']+=1
                    METASTATS['metaWeightUpdate']+=1
                if 'AWE' in weightType:
                    modelSet = weights['AWEmodelSet']
                newTarget = False
            ofUse += 1
            # if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
            if 'OLS' in weightType:
                modelWeights = weights['sourceR2s']
                numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                numModelsUsed.append(numModels)
            # if weightType == 'OLSFE' or weightType =='OLS':
            #    # resultFile = open('performanceVsCost3'+str(ID)+str(weightType)+'.csv','a')
            #    # #wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
            #    # if len(historyData)<30:
            #        # wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
            #    # else:
            #        # wErr = metrics.r2_score(historyData.iloc[(idx-31):][tLabel],historyData.iloc[(idx-31):]['predictions'])
            #    # modelWeights = weights['sourceR2s']
            #    # numModels = sum(1 for val in modelWeights.values() if val != 0)
            #    # resultFile.write(str(idx)+','+str(wErr)+','+str(numModels+1)+'\n')
            #    # resultFile.close()
            # prediction = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,
                    # weightType,stableLocals,modelID,PCs)
            print("modelID:"+str(modelID))
            prediction,numBases,currentBases = modelMulti.instancePredict(df,idx,modelSet,targetModel,weights,tLabel,DROP_FIELDS,
                    weightType,stableLocals,modelID,PCs)
            MODELUSAGE[modelID]+=1
            METASTATS['numberBaseLearners'].append(numBases)
            METASTATS['baseLearnerList'].append(currentBases)
            if not set(METASTATS['currentBaseLearnerIDs']) == set(currentBases):
                METASTATS['baseLearnerChange']+=1
                METASTATS['currentBaseLearnerIDs']=currentBases
            historyData = historyData.append(prediction)
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            p = p.append(targetPred) 
            if idx%50 == 0:
                print(idx, weights)
            diff = abs(targetPred['predictions']-targetPred[tLabel])

            if retrainFlag==False:
                updateFlag,n1,n0,u1,u0 = adwin.update(diff)
            else:
                updateFlag = False
                window = window.append(prediction)
                if len(window) >= MAX_WINDOW:
                    lastModID = modelID
                    modelCount+=1
                    startIDX = idx
                    modelStart.append(startIDX)
                    conceptSize = len(window)
                    partial = window.copy()
                    #retrain model
                    print("RtR: "+str(idx))
                    # sourceModels=readyToReceive(s,sourceModels,ID)
                    sourceModels,METASTATS=readyToReceive(s,sourceModels,ID,METASTATS)
                    modelID, existingModels,transitionMatrix,modelOrder,targetModel,newTarget = modelMultiHistory.nextModels(
                            existingModels,transitionMatrix,modelOrder,partial,modelID,tLabel,DROP_FIELDS,startIDX,False,
                            weightType,PCAVAR,DEFAULT_PRED)
                    if 'AWE' in weightType:
                        modelSet = AWEaddToModelSet(partial,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS)
                    elif 'AddExpO' in weightType:
                        print(lastModID)
                        print(modelSet)
                        modelSet,orderedModels,weights['metaModel'] = AddExpOaddToModelSet(partial,idx,lastModID,existingModels,
                                sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS)
                        METASTATS['metaWeightUpdate']+=1
                        print("after")
                        print(modelSet)
                        print(modelMultiHistory.isStable(lastModID,existingModels))
                    elif 'AddExpP' in weightType:
                        print(lastModID)
                        print(modelSet)
                        modelSet,orderedModels,weights['metaModel'] = AddExpPaddToModelSet(partial,idx,lastModID,existingModels,
                                sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS)
                        METASTATS['metaWeightUpdate']+=1
                        print("after")
                        print(modelSet)
                        print(modelMultiHistory.isStable(lastModID,existingModels))
                        # modelSet,orderedModels,weights['metaModel'] = AddExpPaddToModelSet(df,None,None,sourceModels,
                                # modelSet,orderedModels,weights['metaModel'],tLabel,DROP_FIELDS)
                    else:
                        modelSet,sourceModels,distanceMatrix,tc = addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,distanceMatrix)
                        METASTATS['totalPACalcs']+=tc
                        if tc > 0:
                            METASTATS['triggerPACalcs']+=1
                        # modelSet = addToModelSet(lastModID,existingModels,sourceModels,modelSet)
                    # modelSet = addToModelSet(lastModID,existingModels,sourceModels,modelSet)
                    # if 'AWE' in weightType:
                        # modelSet = updateAWEModelSet(partial,targetModel,modelID,modelSet,META_SIZE,tLabel,DROP_FIELDS)
                    # send to controller
                    ofUse = 0
                    sentFlag = 0
                    # stableLocals = modelMultiHistory.getStableModels(existingModels)
                    # weights = modelMulti.calcWeights(partial,sourceModels,targetModel,tLabel,
                            # DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,existingModels[modelID]['PCs'])
                    # weights,distanceMatrix,affinityMatrix,groupedNames = modelMulti.calcWeights(partial,sourceModels,
                            # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                            # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames)
                    print("in this else")
                    print(modelID)
                    print("distancematrix in this part")
                    print(distanceMatrix)
                    if 'AddExp' not in weightType:
                        weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(partial,modelSet,
                                targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,METASTATS['currentBaseLearnerIDs'])
                        if 'PAC' in weightType or 'CL2' in weightType:
                            METASTATS['totalPACalcs']+=tc[1]
                            if tc[1] > 0:
                                METASTATS['triggerPACalcs']+=1
                            METASTATS['totalOtherMetricCalcs']+=tc[0]
                            if tc[0]>0:
                                METASTATS['triggerMetricCalcs']+=1
                        else:
                            METASTATS['totalOtherMetricCalcs']+=tc
                            if tc>0:
                                METASTATS['triggerMetricCalcs']+=1
                        METASTATS['metaWeightUpdate']+=1
                    if 'AWE' in weightType:
                        modelSet = weights['AWEmodelSet']
                    newTarget = False
                    adwin=Adwin(MAX_WINDOW,ADWIN_DELTA)
                    # window = createModel.initialPredict(window,targetModel,tLabel,DROP_FIELDS)
                    if lastModID < modelID:
                        drifts[modelCount] = week
                    
                    # startDate=idx
                    # modelID,existingModels,transitionMatrix,modelOrder,targetModel,sourceAlpha = gotlHistory.nextModels(existingModels,transitionMatrix,modelOrder,partial,
                                # modelID,sourceAlpha,DELTA,tLabel,DROP_FIELDS,startDate,False)
                    # #create new adwin
                    # adwin=Adwin(MAX_WINDOW,ADWIN_DELTA)
                    retrainFlag=False
                    window = window.iloc[0:0]
            if updateFlag:
                retrainFlag = True
                print("change has been detected at: "+str(idx))
                print("n1, n0: "+str(n1)+", "+str(n0))
                print("u1, u0: "+str(u1)+", "+str(u0))
                if n1 >= MAX_WINDOW:
                    lastModID = modelID
                    modelCount+=1
                    startIDX = idx
                    modelStart.append(startIDX)
                    conceptSize = len(window)
                    # partial = window.copy()
                    partial = historyData.iloc[(MAX_WINDOW*(-1)):].copy()
                    #retrain model
                    print("new model if n1 >= window")
                    print(partial)
                    print("len of window is: "+str(len(partial)))
                    print("RtR: "+str(idx))
                    # sourceModels=readyToReceive(s,sourceModels,ID)
                    sourceModels,METASTATS=readyToReceive(s,sourceModels,ID,METASTATS)
                    modelID, existingModels,transitionMatrix,modelOrder,targetModel,newTarget = modelMultiHistory.nextModels(
                            existingModels,transitionMatrix,modelOrder,partial,modelID,tLabel,DROP_FIELDS,startIDX,False,
                            weightType,PCAVAR,DEFAULT_PRED)
                    # send to controller
                    if 'AWE' in weightType:
                        modelSet = AWEaddToModelSet(partial,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS)
                    elif 'AddExpO' in weightType:
                        print("id")
                        print(lastModID)
                        print(modelSet)
                        modelSet,orderedModels,weights['metaModel'] = AddExpOaddToModelSet(partial,idx,lastModID,existingModels,
                                sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS)
                        METASTATS['metaWeightUpdate']+=1
                        print("after")
                        print(modelSet)
                        print(modelMultiHistory.isStable(lastModID,existingModels))
                    elif 'AddExpP' in weightType:
                        print("id")
                        print(lastModID)
                        print(modelSet)
                        modelSet,orderedModels,weights['metaModel'] = AddExpPaddToModelSet(partial,idx,lastModID,existingModels,
                                sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS)
                        METASTATS['metaWeightUpdate']+=1
                        print("after")
                        print(modelSet)
                        print(modelMultiHistory.isStable(lastModID,existingModels))
                    else:
                        modelSet,sourceModels,distanceMatrix,tc = addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,distanceMatrix)
                        METASTATS['totalPACalcs']+=tc
                        if tc > 0:
                            METASTATS['triggerPACalcs']+=1
                        # modelSet = addToModelSet(lastModID,existingModels,sourceModels,modelSet)
                    # modelSet = addToModelSet(lastModID,existingModels,sourceModels,modelSet)
                    # if 'AWE' in weightType:
                        # modelSet = updateAWEModelSet(partial,targetModel,modelID,modelSet,META_SIZE,tLabel,DROP_FIELDS)
                    ofUse = 0
                    sentFlag = 0
                    # stableLocals = modelMultiHistory.getStableModels(existingModels)
                    # weights = modelMulti.calcWeights(partial,sourceModels,targetModel,tLabel,
                            # DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,existingModels[modelID]['PCs'])
                    # weights,distanceMatrix,affinityMatrix,groupedNames = modelMulti.calcWeights(partial,sourceModels,
                            # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                            # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames)
                    if 'AddExp' not in weightType:
                        weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(partial,modelSet,
                                targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,METASTATS['currentBaseLearnerIDs'])
                        if 'PAC' in weightType or 'CL2' in weightType:
                            METASTATS['totalPACalcs']+=tc[1]
                            if tc[1] > 0:
                                METASTATS['triggerPACalcs']+=1
                            METASTATS['totalOtherMetricCalcs']+=tc[0]
                            if tc[0]>0:
                                METASTATS['triggerMetricCalcs']+=1
                        else:
                            METASTATS['totalOtherMetricCalcs']+=tc
                            if tc>0:
                                METASTATS['triggerMetricCalcs']+=1
                        METASTATS['metaWeightUpdate']+=1
                    if 'AWE' in weightType:
                        modelSet = weights['AWEmodelSet']
                    newTarget = False
                    adwin=Adwin(MAX_WINDOW,ADWIN_DELTA)
                    if lastModID < modelID:
                        drifts[modelCount] = week
                    
                    retrainFlag=False
                    window = window.iloc[0:0]
                else:
                    window = historyData.iloc[int(n1)*-1:].copy()
        # if idx%75 == (startDoW-1) and storeFlag == True:
        for b in METASTATS['currentBaseLearnerIDs']:
            if b in METASTATS['modelsReceived']:
                if b not in METASTATS['baseUsage'].keys():
                    METASTATS['baseUsage'][b]=1
                else:
                    METASTATS['baseUsage'][b]+=1
        if idx%(int(MAX_WINDOW/2)) == (startDoW-1) and storeFlag == True:
            storeFlag = False
            w = dict()
            for k in weights['sourceR2s']:
                w[k] = weights['sourceR2s'][k]/weights['totalR2']
            multiWeights.append(w)
    endIDX = historyData.index.max()
    modelOrder,existingModels = modelMultiHistory.updateLastModelUsage(modelOrder,existingModels,endIDX)
    print("number concepts: "+str(modelCount))
    print(modelStart)
    print(modelOrder)
    print(existingModels)
    print(transitionMatrix)
    mse = metrics.mean_squared_error(historyData[tLabel],historyData['predictions'])
    print(mse)
    print(calcError(historyData[tLabel],historyData['predictions']))
    w = dict()
    for k in weights['sourceR2s']:
        w[k] = weights['sourceR2s'][k]/weights['totalR2']
    multiWeights.append(w)
    print("all weights are:")
    print(multiWeights)
    print(metrics.r2_score(p[tLabel],p['predictions']))
    print(sourceModels)
    print("modelset")
    print(modelSet)

    pd.set_option('display.max_rows',None)
    print ("predictions")
    print(historyData[[tLabel,'predictions']])
    print(metrics.r2_score(historyData.iloc[81:][tLabel],historyData.iloc[81:]['predictions']))


    return historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights,p,numModelsUsed



def simTarget(weightType,s):
    global METASTATS
    sourceModels = dict()
    sourceModels,METASTATS = readyToReceive(s,sourceModels,ID,METASTATS)
    # sourceModels = dict()
    # sourceModels = readyToReceive(s,sourceModels,ID)

    #get models from controller
    print("source Models:")
    print(sourceModels)
    if DEFAULT_PRED == 'Following':
        df = preprocess.pullFollowingData(FP)
        df = preprocess.subsetFollowingData(df,SIM_FROM,SIM_TO)
    elif DEFAULT_PRED == 'Heating':
        df = preprocess.pullHeatingData(FP)
        df = preprocess.subsetHeatingData(df,SIM_FROM,SIM_TO)
    else:
        df = preprocess.pullData(FP)
        df = preprocess.subsetData(df,SIM_FROM,SIM_TO)
    df['predictions'] = np.nan
    
    historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights,p,numModelsUsed = runADWIN(df,sourceModels,tLabel,weightType,s)
    
    numModelsUsedArr = np.array(numModelsUsed)
    meanNumModels = numModelsUsedArr.mean()
    print("stablesixe: "+str(STABLE_SIZE))
    print(" maxwindow: "+str(MAX_WINDOW))
    print("adwinDelta: "+str(ADWIN_DELTA))
    if WEIGHTTYPE == 'OLS':
        fileWeight = 'OLS'
    elif WEIGHTTYPE == 'OLSFEMI':
        fileWeight = str(WEIGHTTYPE)+str(MITHRESH)+"_"+str(CULLTHRESH)
    elif WEIGHTTYPE == 'OLSFEPA':
        fileWeight = str(WEIGHTTYPE)+str(PATHRESH)+"_"+str(CULLTHRESH)
    elif WEIGHTTYPE == 'OLSPARed':
        fileWeight = 'OLSPARed'
    elif WEIGHTTYPE == 'OLSKPAC2Red':
        fileWeight = 'OLSKPAC2Red'
    elif WEIGHTTYPE == 'OLSCL':
        fileWeight = 'OLSCL'
    elif WEIGHTTYPE == 'OLSCL2':
        fileWeight = 'OLSCL2'
    elif WEIGHTTYPE == 'OLSPAC':
        fileWeight = 'OLSPAC'
    elif WEIGHTTYPE == 'OLSPAC2':
        fileWeight = 'OLSPAC2'
    elif WEIGHTTYPE == 'OLSKPAC':
        fileWeight = 'OLSKPAC'
    elif WEIGHTTYPE == 'OLSKPAC2':
        fileWeight = 'OLSKPAC2'
    elif WEIGHTTYPE == 'OLSAWE':
        fileWeight = 'OLSAWE'
    elif WEIGHTTYPE == 'OLSAddExpO':
        fileWeight = 'OLSAddExpO'
    elif WEIGHTTYPE == 'OLSAddExpP':
        fileWeight = 'OLSAddExpP'
    else:
        fileWeight = str(WEIGHTTYPE)+str(CULLTHRESH)
    
    if not os.path.exists('Results/ADWIN/'+str(DEFAULT_PRED)):
        os.mkdir('Results/ADWIN/'+str(DEFAULT_PRED))
    if not os.path.exists('Results/ADWIN/'+str(DEFAULT_PRED)+'/'+str(fileWeight)):
        os.mkdir('Results/ADWIN/'+str(DEFAULT_PRED)+'/'+str(fileWeight))
    if not os.path.isfile('Results/ADWIN/'+str(DEFAULT_PRED)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv"):
        resFile = open('Results/ADWIN/'+str(DEFAULT_PRED)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
        # resFile.write('sID,R2,RMSE,Pearsons,PearsonsPval,Err,numModels,reproR2,reproRMSE,reproPearsons,reproPearsonsPval,reproErr,numModelsUsed\n')
        # resFile.write('sID,Win,Thresh,R2,RMSE,Err,Pearsons,PearsonsPval,numModels,reproR2,reproRMSE,'+
                # 'reproErr,reproPearsons,reproPearsonsPval,numModelsUsed,totalLocalModels,totalStableLocalModels,'+
                # 'METAnumModelsSent,METAstableLocals,METAnumModelsRec,METABaseChange,METABaseUsage,METAAvgBases,METAWeightUpdates\n')
        resFile.write('sID,Win,AdwinDelta,R2,RMSE,Err,Pearsons,PearsonsPval,numModels,reproR2,reproRMSE,'+
                'reproErr,reproPearsons,reproPearsonsPval,numModelsUsed,totalLocalModels,totalStableLocalModels,'+
                'METAnumModelsSent,METAstableLocals,METAnumModelsRec,METABaseChange,METABaseUsage,METAAvgBases,METAWeightUpdates,'+
                'totalPACalcs,totalOtherMetricCalcs,triggerPACalcs,triggerMetricCalcs\n')
    else:
        resFile = open('Results/ADWIN/'+str(DEFAULT_PRED)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
    # resFile.write(str(UID)+','+str(MAX_WINDOW)+','+str(THRESHOLD)+','+
    resFile.write(str(UID)+','+str(MAX_WINDOW)+','+str(ADWIN_DELTA)+','+
            str(metrics.r2_score(historyData[tLabel],historyData['predictions']))+','+
            str(np.sqrt(metrics.mean_squared_error(historyData[tLabel],historyData['predictions'])))+','+
            str(metrics.mean_absolute_error(historyData[tLabel],historyData['predictions']))+','+
            str(pearsonr(historyData['predictions'].values,historyData[tLabel].values)[0])+','+
            str(pearsonr(historyData['predictions'].values,historyData[tLabel].values)[1])+','+
            str(len(sourceModels))+','+
            str(metrics.r2_score(p[tLabel],p['predictions']))+','+
            str(np.sqrt(metrics.mean_squared_error(p[tLabel],p['predictions'])))+','+
            str(metrics.mean_absolute_error(p[tLabel],p['predictions']))+',' +
            str(pearsonr(p['predictions'].values,p[tLabel].values)[0])+','+
            str(pearsonr(p['predictions'].values,p[tLabel].values)[1])+','+
            str(meanNumModels)+','+str(len(existingModels))+','+
            str(len(modelMultiHistory.getStableModels(existingModels)))+','+
            str(len(METASTATS['modelsSent']))+','+str(len(METASTATS['stableLocal']))+','+
            str(len(METASTATS['modelsReceived']))+','+str(METASTATS['baseLearnerChange'])+','+
            '\"'+str(METASTATS['baseUsage'])+'\",'+
            str(sum(METASTATS['numberBaseLearners'])/len(METASTATS['numberBaseLearners']))+','+
            str(METASTATS['metaWeightUpdate'])+','+
            str(METASTATS['totalPACalcs'])+','+
            str(METASTATS['totalOtherMetricCalcs'])+','+
            str(METASTATS['triggerPACalcs'])+','+
            str(METASTATS['triggerMetricCalcs'])+','+'\n')
            # str(len(modelMultiHistory.getStableModels(existingModels)))+','+
            # str(len(METASTATS['modelsSent']))+','+str(len(METASTATS['stableLocal']))+','+
            # str(len(METASTATS['modelsReceived']))+','+str(METASTATS['baseLearnerChange'])+','+
            # '\"'+str(METASTATS['baseUsage'])+'\",'+
            # str(sum(METASTATS['numberBaseLearners'])/len(METASTATS['numberBaseLearners']))+','+
            # str(METASTATS['metaWeightUpdate'])+'\n')

    # resFile.write("METASTATS stable local: "+str(METASTATS['stableLocal'])+'\n'+
        # "METASTATS modelsSent: "+str(METASTATS['modelsSent'])+'\n'+
        # "METASTATS modelsReceived: "+str(METASTATS['modelsReceived'])+'\n'+
        # "METASTATS baseLearnerChange: "+str(METASTATS['baseLearnerChange'])+'\n'+
        # "METASTATS baseLearnerList: "+str(METASTATS['baseLearnerList'])+'\n'+
        # "METASTATS baseUsage: "+str(METASTATS['baseUsage'])+'\n'+
        # "METASTATS numberBaseLearners: "+str(METASTATS['numberBaseLearners'])+'\n'+
        # "METASTATS totalNumInstances: "+str(METASTATS['totalNumInstances'])+'\n'+
        # "\n\nAVERAGING\n\n"+
        # "total number models sent: "+str(len(METASTATS['modelsSent']))+'\n'+
        # "total number models stable: "+str(len(METASTATS['stableLocal']))+'\n'+
        # "total number models received: "+str(len(METASTATS['modelsReceived']))+'\n'+
        # "total average num base learners: "+str(sum(METASTATS['numberBaseLearners'])/len(METASTATS['numberBaseLearners']))+'\n'+
        # "number of nonused models: "+str(sum(b==0 for b in METASTATS['baseUsage'].values()))+'\n'+
        # "total weight updates: "+str(METASTATS['metaWeightUpdate'])+'\n'+
        # "PERFORMANCE: "+ str(metrics.r2_score(historyData[tLabel],historyData['predictions'])))
    resFile.close()

    
    # if not os.path.exists('Results/ADWIN/'+str(DEFAULT_PRED)):
        # os.mkdir('Results/ADWIN/'+str(DEFAULT_PRED))
    # if not os.path.exists('Results/ADWIN/'+str(DEFAULT_PRED)+'/'+str(fileWeight)):
        # os.mkdir('Results/ADWIN/'+str(DEFAULT_PRED)+'/'+str(fileWeight))
    # if not os.path.isfile('Results/ADWIN/'+str(DEFAULT_PRED)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv"):
        # resFile = open('Results/ADWIN/'+str(DEFAULT_PRED)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
        # resFile.write('sID,Win,AdwinDelta,R2,RMSE,Err,Pearsons,PearsonsPval,numModels,reproR2,reproRMSE,reproErr,reproPearsons,reproPearsonsPval,numModelsUsed,totalLocalModels,totalStableLocalModels\n')
    # else:
        # resFile = open('Results/ADWIN/'+str(DEFAULT_PRED)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
    # resFile.write(str(UID)+','+str(MAX_WINDOW)+','+str(ADWIN_DELTA)+','+
            # str(metrics.r2_score(historyData[tLabel],historyData['predictions']))+','+
            # str(np.sqrt(metrics.mean_squared_error(historyData[tLabel],historyData['predictions'])))+','+
            # str(metrics.mean_absolute_error(historyData[tLabel],historyData['predictions']))+','+
            # str(pearsonr(historyData['predictions'].values,historyData[tLabel].values)[0])+','+
            # str(pearsonr(historyData['predictions'].values,historyData[tLabel].values)[1])+','+
            # str(len(sourceModels))+','+
            # str(metrics.r2_score(p[tLabel],p['predictions']))+','+
            # str(np.sqrt(metrics.mean_squared_error(p[tLabel],p['predictions'])))+','+
            # str(metrics.mean_absolute_error(p[tLabel],p['predictions']))+',' +
            # str(pearsonr(p['predictions'].values,p[tLabel].values)[0])+','+
            # str(pearsonr(p['predictions'].values,p[tLabel].values)[1])+','+
            # str(meanNumModels)+','+str(len(existingModels))+','+
            # str(len(modelMultiHistory.getStableModels(existingModels)))+'\n')

    # resFile.close()


    return historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights

def main():
    global ID
    global PORT
    global SIM_FROM
    global SIM_TO
    global FP
    global INIT_DAYS
    global STABLE_SIZE
    global MAX_WINDOW
    global ADWIN_DELTA
    global RUNID
    global NUMSTREAMS
    global WEIGHTTYPE
    global UID
    global CULLTHRESH
    global MITHRESH
    global DEFAULT_PRED
    global DROP_FIELDS
    global tLabel
    global EPSILON
    global PATHRESH
    global PCAVAR
    global METASTATS

    #loqxy
    parser = OptionParser(usage="usage: prog options",version="BOTL v2.0")
    parser.add_option("-k","--id",default="source0",dest="ID",help="name of source: sourceID")
    parser.add_option("-u","--uid",default="0",dest="UID",help="ID number of source")
    parser.add_option("-b","--port",default="3000",dest="PORT",help="PORT")
    parser.add_option("-f","--fp",default="./",dest="FP",help="file path")
    parser.add_option("-g","--from",default="0",dest="FROM",help="simulation from")
    parser.add_option("-j","--to",default="15000",dest="TO",help="simulation to")
    parser.add_option("-d","--domain",default = "Following",dest="DEFAULT_PRED",help="domain: Following, Heating, Sudden, Gradual")
    # parser.add_option("-t","--type",default = "RePro",dest= "CD_TYPE",help="Concept Drift Type: RePro, ADWIN, AWPro")
    parser.add_option("-w","--window",default = "90",dest="MAX_WINDOW",help="Window size (default = 90)")
    parser.add_option("-r","--ReProAcc",default = "0.5",dest="MODEL_HIST_THRESHOLD_ACC",help="RePro drift threshold")
    parser.add_option("-p","--ReProProb",default = "0.5",dest="MODEL_HIST_THRESHOLD_PROB",help="RePro recur prob")
    parser.add_option("-i","--runid",default = "1",dest="runID",help="RunID")
    parser.add_option("-n","--numStreams",default = "1",dest="numStreams",help="Number of streams")
    parser.add_option("-z","--ADWINDelta",default = "0.02",dest="ADWIN_DELTA",help="ADWIN confidence value")
    # parser.add_option("-e","--ReProThresh",default = "0.1",dest="THRESHOLD",help="RePro error threshold")
    parser.add_option("-s","--socket",default = "3000",dest="socketOffset",help="Socket Offset")
    parser.add_option("-e","--ensemble",default = "OLS",dest="weightType",help="Weight Type (OLS, OLSFE, OLSFEMI,...)")
    parser.add_option("-c","--perfCull",default = "0.0",dest="CThresh",help="Performance culling parameter")
    parser.add_option("-m","--miCull",default = "2",dest="MThresh",help="Mutual Information culling parameter")
    parser.add_option("-a","--paCull",default = "1",dest="PAThresh",help="Principal Angle culling parameter")
    parser.add_option("-v","--variance",default = "0.05",dest="PCAVar",help="Keep prinicpal components that capture this uch variance")
    (options,args) = parser.parse_args()
    
    HOST = 'localhost'
    ID = str(options.ID)
    UID = str(options.UID)
    PORT = int(options.PORT)
    FP = str(options.FP)
    DEFAULT_PRED=str(options.DEFAULT_PRED)
    # CD_TYPE=str(options.CD_TYPE)
    MAX_WINDOW=int(options.MAX_WINDOW)
    MODEL_HIST_THRESHOLD_ACC=float(options.MODEL_HIST_THRESHOLD_ACC)
    MODEL_HIST_THRESHOLD_PROB=float(options.MODEL_HIST_THRESHOLD_PROB)
    RUNID=int(options.runID)
    NUMSTREAMS=int(options.numStreams)
    ADWIN_DELTA=float(options.ADWIN_DELTA)
    socketOffset=int(options.socketOffset)
    WEIGHTTYPE=str(options.weightType)
    CULLTHRESH=float(options.CThresh)
    MITHRESH=float(options.MThresh)
    PATHRESH=float(options.PAThresh)
    PCAVAR=float(options.PCAVar)
    THRESHOLD = MODEL_HIST_THRESHOLD_ACC
    STABLE_SIZE = MAX_WINDOW * 2
    INIT_DAYS = MAX_WINDOW

    # HOST = 'localhost'
    # ID = str(sys.argv[1])
    # PORT = int(sys.argv[2])
    # # SIM_FROM = int(sys.argv[3])
    # # SIM_TO = int(sys.argv[4])
    # FP = str(sys.argv[5])
    # INIT_DAYS = int(sys.argv[6])
    # STABLE_SIZE = int(sys.argv[7])
    # MAX_WINDOW = int(sys.argv[8])
    # ADWIN_DELTA = float(sys.argv[9])
    # weightType = str(sys.argv[10])
    # RUNID = int(sys.argv[11])
    # NUMSTREAMS = int(sys.argv[12])
    # UID = str(sys.argv[13])
    # WEIGHTTYPE = weightType
    # CULLTHRESH = float(sys.argv[14])
    # MITHRESH = float(sys.argv[15])
    # DEFAULT_PRED = str(sys.argv[18])
    # if DEFAULT_PRED == 'heating':
        # SIM_FROM = datetime.strptime(sys.argv[3],"%Y-%m-%d").date()
        # SIM_TO = datetime.strptime(sys.argv[4],"%Y-%m-%d").date()
    # else:
        # SIM_FROM = int(sys.argv[3])
        # SIM_TO = int(sys.argv[4])
    if DEFAULT_PRED == 'Heating':
        # SIM_FROM = datetime.strptime(sys.argv[3],"%Y-%m-%d").date()
        # SIM_TO = datetime.strptime(sys.argv[4],"%Y-%m-%d").date()
        SIM_FROM = datetime.strptime(options.FROM,"%Y-%m-%d").date()
        SIM_TO = datetime.strptime(options.TO,"%Y-%m-%d").date()
    else:
        # SIM_FROM = int(sys.argv[3])
        # SIM_TO = int(sys.argv[4])
        SIM_FROM = int(options.FROM)
        SIM_TO = int(options.TO)

    if DEFAULT_PRED == 'Following':
        DROP_FIELDS = ['Time','predictions']
        tLabel='FollowingTime'
        EPSILON = 0.1
    elif DEFAULT_PRED == 'Heating':
        DROP_FIELDS = ['date','year','month','day','predictions','heatingOn']
        tLabel='dTemp'
        EPSILON = 0.5
    else:
        DROP_FIELDS = ['predictions']
        tLabel='y'
        EPSILON = 0.01
    
    # METASTATS = {'stableLocal':list(),'modelsSent':list(),'modelsReceived':list(),'metaWeightUpdate':0,'baseLearnerChange':0,
            # 'baseLearnerList':list(),'currentBaseLearnerIDs':list(),'baseUsage':dict(),'numberBaseLearners':list(),'totalNumInstances':0}
    METASTATS = {'stableLocal':list(),'modelsSent':list(),'modelsReceived':list(),'metaWeightUpdate':0,'baseLearnerChange':0,
            'baseLearnerList':list(),'currentBaseLearnerIDs':list(),'baseUsage':dict(),'numberBaseLearners':list(),'totalNumInstances':0,
            'totalPACalcs':0,'totalOtherMetricCalcs':0,'triggerPACalcs':0,'triggerMetricCalcs':0}

    
    '''
    # set up connection to controller #
    '''

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST,PORT))
    s.sendall(ID.encode())
    print("connection established")
    print(ID, PORT, SIM_FROM, SIM_TO, FP, INIT_DAYS, STABLE_SIZE, MAX_WINDOW, ADWIN_DELTA)
    print("stablesixe: "+str(STABLE_SIZE))
    print(" maxwindow: "+str(MAX_WINDOW))
    print("adwin_delta: "+str(ADWIN_DELTA))

    ack = s.recv(1024).decode()
    print(repr(ack))
    simTarget(WEIGHTTYPE,s)

    s.close()


if __name__ == '__main__':main()

