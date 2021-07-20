import numpy as np
import subprocess
import socket
import random
# import source
import pandas as pd
import pickle
import threading
import time
import sys
import os
from optparse import OptionParser

MODELS = dict()
INIT_DAYS = 0#80
MODEL_HIST_THRESHOLD_PROB = 0# 0.4
MAX_WINDOW = 0#80
STABLE_SIZE = 0#2* MAX_WINDOW
MODEL_HIST_THRESHOLD_ACC = 0#0.5
THRESHOLD = 0#0.5
DEFAULT_PRED = ''
CD_TYPE = ''

class myThread (threading.Thread):
    def __init__(self,threadID,info,receivedModels,runnum,nums):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = info['Name']
        self.uid = info['uid']
        self.outputFile = info['stdo']
        self.PORT = info['PORT']
        self.fp = info['Run']
        self.inputFile = info['stdin']
        self.sFrom = info['sFrom']
        self.sTo = info['sTo']
        self.weightType = info['weightType']
        self.cullThresh = info['cullThresh']
        self.miThresh = info['miThresh']
        self.receivedModels = receivedModels
        self.runID = runnum
        self.numStreams = nums
        self.default_pred = info['default_pred']
        self.PCAVar = info['PCAVar']
        self.paThresh = info['paThresh']
    
    def run(self):
        print("starting "+ self.name)
        initiate(self.threadID,self.name,self.uid,self.PORT,self.fp,self.inputFile,self.outputFile,self.
                sFrom,self.sTo,self.weightType,self.receivedModels,self.runID,self.numStreams,
                self.cullThresh,self.miThresh,self.paThresh,self.PCAVar,self.default_pred)
        print("exiting " + self.name)

def getModelsToSend(threadID,modelsSent):
    toSend = dict()
    allModels = MODELS
    for tID,modelDict in allModels.items():
        if tID != threadID:
            for modelID,model in modelDict.items():
                sourceModID = str(tID)+'-'+str(modelID)
                print("sourceModID: "+str(sourceModID))
                print("modelsSent: "+str(modelsSent))
                if sourceModID not in modelsSent:
                    toSend[sourceModID] = model
    return toSend


def sendHandshake(targetID,data,conn,modelsSent):
    RTRFlag = data.split(',')[0]
    if RTRFlag == 'RTR':
        target_ID = int(data.split(',')[1])
        if target_ID != targetID:
            print("changed targetIDs")
            return 0,modelsSent
        #get number of models to send
        modelsToSend = getModelsToSend(targetID,modelsSent)
        numModels = len(modelsToSend)
        conn.sendall(('ACK,'+str(numModels)).encode())
        ack = conn.recv(1024).decode()
        print(targetID, repr(ack))
        if ack == 'ACK':
            return sendModels(targetID,numModels,modelsToSend,modelsSent,conn)
        elif ack == 'END':
            return 1,modelsSent
    return 0,modelsSent

def sendModels(targetID,numModels,modelsToSend,modelsSent,conn):
    for modelID,model in modelsToSend.items():
        modelToSend = pickle.dumps(model)
        print("modelID is: "+str(modelID))
        
        brokenBytes = [modelToSend[i:i+1024] for i in range(0,len(modelToSend),1024)]
        numPackets = len(brokenBytes)
        lenofModel = len(modelToSend)
        print(str(targetID)+"BROKEN BYTES LEN: "+str(numPackets))
        conn.sendall(('RTS,'+str(modelID)+','+str(numPackets)+','+str(lenofModel)).encode())
        ack = conn.recv(1024).decode()
        ackNumPackets = int(ack.split(',')[1])
        print("acked number of packets is " +str(ackNumPackets))

        if ackNumPackets == numPackets:
            flag,modelsSent = sendModel(modelID,brokenBytes,modelsSent,conn)
        else:
            print("failed to send model: "+str(modelID))
            return 0,modelsSent
    return 1,modelsSent

def sendModel(modelID,brokenBytes,modelsSent,conn):
    for idx,i in enumerate(brokenBytes):
        conn.sendall(i)
    recACK = conn.recv(1024).decode()
    if modelID == recACK.split(',')[1]:
        modelsSent.append(modelID)
        print("models sent: "+str(modelsSent))
        return 1, modelsSent
    return 0, modelsSent


def receiveHandshake(sourceID,data,conn):
    RTSFlag = data.split(',')[0]
    if RTSFlag == 'RTS':
        modelID = data.split(',')[1]
        print(modelID)
        numPackets = int(data.split(',')[2])
        lenofModel = int(data.split(',')[3])

        conn.sendall(('ACK,'+str(numPackets)).encode())

        return receiveData(sourceID,modelID,numPackets,lenofModel,conn)
        
    return 0

def receiveData(sourceID,modelID,numPackets,lenofModel,conn):
    # send ACK
    pickledModel = b''
    while (len(pickledModel)<lenofModel):
        pickledModel = pickledModel + conn.recv(1024)
    conn.sendall(('RECEIVED,'+str(modelID)).encode())

    storeModel(sourceID,modelID,pickledModel)
    return 1

def storeModel(sourceID,modelID,pickledModel):
    global MODELS
    model = pickle.loads(pickledModel)
    print(sourceID, modelID)
    MODELS[sourceID][modelID] = model

    # print(sourceID, MODELS[sourceID])

def initiate(threadID,name,uid,PORT,fp,inFile,outFile,sFrom,sTo,weightType,recievedModels,
        runID,nums,cullThresh,miThresh,paThresh,PCAVar,default_pred):
    out = open(os.devnull,'w')
    # if ('PAC' in weightType) or weightType == 'OLSPAC' or weightType == 'OLSCL2' or weightType == 'OLSCL' or 'AWE' in weightType or 'AddExp' in weightType:
        # out = open(outFile,'w')
    # else:
        # out = open(os.devnull,'w')
    # out = open(os.devnull,'w')
    # if ('D004J006' in uid) or ('D003J005' in uid) or ('SuddenT2' in uid):
        # out = open(outFile,'w')
    # out = open(outFile,'w')

    modelsSent = []
    if CD_TYPE == 'RePro':
        args = ["python3",fp,
                "--id", str(threadID),
                "--port",str(PORT),
                "--from",str(sFrom),
                "--to",str(sTo),
                "--fp",str(inFile),
                "--window",str(MAX_WINDOW),
                "--ReProAcc",str(MODEL_HIST_THRESHOLD_ACC),
                "--ReProProb",str(MODEL_HIST_THRESHOLD_PROB),
                "--ensemble",str(weightType),
                "--runid",str(runID),
                "--numStreams",str(nums),
                "--uid",str(uid),
                "--perfCull",str(cullThresh),
                "--miCull",str(miThresh),
                "--paCull",str(paThresh),
                "--variance",str(PCAVar),
                "--domain",str(default_pred)]
        # args = ['python3',fp,str(threadID),str(PORT),str(sFrom),str(sTo),inFile,str(INIT_DAYS),str(MODEL_HIST_THRESHOLD_ACC), 
                # str(MODEL_HIST_THRESHOLD_PROB),str(STABLE_SIZE),str(MAX_WINDOW),str(THRESHOLD),str(weightType),str(runID),
                # str(nums),str(uid),str(cullThresh),str(miThresh),str(paThresh),str(PCAVar),str(default_pred)]
        # elif CD_TYPE == 'AWPro':
    else:
        args = ["python3",fp,
                "--id", str(threadID),
                "--port",str(PORT),
                "--from",str(sFrom),
                "--to",str(sTo),
                "--fp",str(inFile),
                "--window",str(MAX_WINDOW),
                "--ReProAcc",str(MODEL_HIST_THRESHOLD_ACC),
                "--ReProProb",str(MODEL_HIST_THRESHOLD_PROB),
                "--ADWINDelta",str(ADWIN_DELTA),
                "--ensemble",str(weightType),
                "--runid",str(runID),
                "--numStreams",str(nums),
                "--uid",str(uid),
                "--perfCull",str(cullThresh),
                "--miCull",str(miThresh),
                "--paCull",str(paThresh),
                "--variance",str(PCAVar),
                "--domain",str(default_pred)]
        # args = ['python3',fp,str(threadID),str(PORT),str(sFrom),str(sTo),inFile,str(INIT_DAYS), 
                # str(STABLE_SIZE),str(MAX_WINDOW),str(ADWIN_DELTA),str(weightType),str(runID),
                # str(nums),str(uid),str(cullThresh),str(miThresh),str(MODEL_HIST_THRESHOLD_ACC),
                # str(MODEL_HIST_THRESHOLD_PROB),str(default_pred)]
    p = subprocess.Popen(args,stdout=out)
    try: 
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.bind(('localhost',PORT))
        s.listen(1)
    except socket.error:
        print("Failed to create socket")
        s.close()
        s = None
    if s is None:
        print("exiting")
        sys.exit(1)
    conn,addr = s.accept()
    source_ID = conn.recv(1024).decode()
    print("connected to: "+repr(source_ID))
    conn.sendall(("connected ACK").encode())
    
    while 1:
        # listen for rts
        # do handshake
        # receive model
        data = conn.recv(1024).decode()
        print(repr(data))
        flag = data.split(',')[0]
        if flag == 'RTS':
            successFlag = receiveHandshake(threadID,data,conn)
        elif flag == 'RTR':
            successFlag, modelsSent = sendHandshake(threadID,data,conn,modelsSent)
        else:
            print("flag recieved is not RTR or RTS")
            successFlag = 0

        #send ACK
        print("connection established with: "+repr(data))
        if not data: break
        if not successFlag:
            print("communication FAIL")
            break
        time.sleep(1)
        
    p.wait()
    conn.close()
    s.close()
    out.close()

def getHeatingDates():
    dates = dict()
    dates = {
            0:{'start':"2014-01-01",'end':"2015-03-31"},
            1:{'start':"2015-01-01",'end':"2015-12-31"},
            2:{'start':"2014-09-01",'end':"2015-03-30"},
            3:{'start':"2015-01-01",'end':"2015-09-30"},
            4:{'start':"2014-01-01",'end':"2015-06-30"}
            }
    # testdates = {
            # 0:{'start':"2014-03-01",'end':"2015-03-31"},
            # 1:{'start':"2015-01-01",'end':"2015-12-31"}}
    # return testdates
    return dates

def getFPdict(driftType):
    FPdict = {
            # (str(driftType)+'S1'):'../../HyperplaneDG/Data/Datastreams/SOURCEMultiConcept'+str(driftType)+'.csv',
            (str(driftType)+'T1'):'HyperplaneSample/TARGET1MultiConcept'+str(driftType)+'1.csv',
            (str(driftType)+'T2'):'HyperplaneSample/TARGET2MultiConcept'+str(driftType)+'1.csv',
            (str(driftType)+'T3'):'HyperplaneSample/TARGET3MultiConcept'+str(driftType)+'1.csv',
            (str(driftType)+'T4'):'HyperplaneSample/TARGET4MultiConcept'+str(driftType)+'1.csv',
            (str(driftType)+'T5'):'HyperplaneSample/TARGET5MultiConcept'+str(driftType)+'1.csv'}
    # testFPdict = {
            # # (str(driftType)+'S1'):'../../HyperplaneDG/Data/Datastreams/SOURCEMultiConcept'+str(driftType)+'.csv',
            # (str(driftType)+'T1'):'../HyperplaneDG/Data/Datastreams/TARGET1MultiConcept'+str(driftType)+'1.csv',
            # (str(driftType)+'T2'):'../HyperplaneDG/Data/Datastreams/TARGET2MultiConcept'+str(driftType)+'1.csv'}
    # return testFPdict
    return FPdict

def get7FollowingFPdict():
    #sample following distance data streams available in Git Repository:XXXXXX
    testFPdict = {
            'D001J001': 'FollowingDistanceSample/dr001J001.csv',
            'D001J002': 'FollowingDistanceSample/dr001J002.csv',
            'D001J003': 'FollowingDistanceSample/dr001J003.csv',
            'D002J001': 'FollowingDistanceSample/dr002J001.csv',
            'D002J002': 'FollowingDistanceSample/dr002J002.csv',
            'D002J003': 'FollowingDistanceSample/dr002J003.csv'}
    return testFPdict


def getFollowingFPdict():
    #files not available for publication
    FPdict = {
            'D001J001': 'FollowingDistanceSample/dr001J001.csv',
            'D001J002': 'FollowingDistanceSample/dr001J002.csv',
            'D001J003': 'FollowingDistanceSample/dr001J003.csv',
            'D002J001': 'FollowingDistanceSample/dr002J001.csv',
            'D002J002': 'FollowingDistanceSample/dr002J002.csv',
            'D002J003': 'FollowingDistanceSample/dr002J003.csv'}
            # 'D001J001': '../FollowingDistanceData/dr001J001.csv',
            # 'D001J002': '../FollowingDistanceData/dr001J002.csv',
            # 'D001J003': '../FollowingDistanceData/dr001J003.csv',
            # 'D002J001': '../FollowingDistanceData/dr002J001.csv',
            # 'D002J002': '../FollowingDistanceData/dr002J002.csv',
            # 'D002J003': '../FollowingDistanceData/dr002J003.csv',
            # # 'D003J001': '../../FollowingDistanceData/dr003J001.csv',
            # 'D003J002': '../FollowingDistanceData/dr003J002.csv',
            # 'D003J005': '../FollowingDistanceData/dr003J005.csv',
            # 'D003J006': '../FollowingDistanceData/dr003J006.csv',
            # 'D004J001': '../FollowingDistanceData/dr004J001.csv',
            # # 'D004J002': '../../FollowingDistanceData/dr004J002.csv',
            # 'D004J003': '../FollowingDistanceData/dr004J003.csv',
            # 'D004J004': '../FollowingDistanceData/dr004J004.csv',
            # 'D004J005': '../FollowingDistanceData/dr004J005.csv',
            # 'D004J006': '../FollowingDistanceData/dr004J006.csv',
            # 'D004J007': '../FollowingDistanceData/dr004J007.csv',
            # # 'D004J008': '../../FollowingDistanceData/dr004J008.csv',
            # # 'D004J016': '../../FollowingDistanceData/dr004J016.csv',
            # # 'D004J017': '../../FollowingDistanceData/dr004J017.csv',
            # 'D004J019': '../FollowingDistanceData/dr004J019.csv',
            # 'D004J020': '../FollowingDistanceData/dr004J020.csv'}#,
            # 'D004J021': '../../FollowingDistanceData/dr004J021.csv'}
    # testFPdict = {
            # 'D001J001': '../FollowingDistanceData/dr001J001.csv',
            # 'D001J003': '../FollowingDistanceData/dr001J003.csv',#}
            # # 'D001J002': '../FollowingDistanceData/dr001J002.csv',
            # # 'D001J003': '../FollowingDistanceData/dr001J003.csv',
            # # 'D002J001': '../FollowingDistanceData/dr002J001.csv',
            # # 'D002J002': '../FollowingDistanceData/dr002J002.csv',
            # # 'D002J003': '../FollowingDistanceData/dr002J003.csv',
            # # 'D003J002': '../FollowingDistanceData/dr003J002.csv',
            # 'D003J005': '../FollowingDistanceData/dr003J005.csv',
            # 'D003J006': '../FollowingDistanceData/dr003J006.csv',
            # 'D004J003': '../FollowingDistanceData/dr004J003.csv',
            # 'D004J005': '../FollowingDistanceData/dr004J005.csv',
            # 'D004J006': '../FollowingDistanceData/dr004J006.csv'}#,
            # # 'D004J007': '../FollowingDistanceData/dr004J007.csv',
            # # 'D004J020': '../FollowingDistanceData/dr004J020.csv',
    # return testFPdict
    return FPdict

def main():
    global MODELS
    global INIT_DAYS#80
    global MODEL_HIST_THRESHOLD_PROB# 0.4
    global MAX_WINDOW#80
    global STABLE_SIZE#2* MAX_WINDOW
    global MODEL_HIST_THRESHOLD_ACC#0.5
    global THRESHOLD#0.5
    global DEFAULT_PRED
    global CD_TYPE
    global ADWIN_DELTA
    parser = OptionParser(usage="usage: prog options",version="BOTL v2.0")
    parser.add_option("-d","--domain",default = "Following",dest="DEFAULT_PRED",help="domain: Following, Heating, Sudden, Gradual")
    parser.add_option("-t","--type",default = "RePro",dest= "CD_TYPE",help="Concept Drift Type: RePro, ADWIN, AWPro")
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
    sourceInfo = dict()
    targetInfo = dict()
    (options,args) = parser.parse_args()
    
    DEFAULT_PRED=str(options.DEFAULT_PRED)
    CD_TYPE=str(options.CD_TYPE)
    MAX_WINDOW=int(options.MAX_WINDOW)
    MODEL_HIST_THRESHOLD_ACC=float(options.MODEL_HIST_THRESHOLD_ACC)
    MODEL_HIST_THRESHOLD_PROB=float(options.MODEL_HIST_THRESHOLD_PROB)
    runID=int(options.runID)
    numStreams=int(options.numStreams)
    ADWIN_DELTA=float(options.ADWIN_DELTA)
    socketOffset=int(options.socketOffset)
    weightType=str(options.weightType)
    CThresh=float(options.CThresh)
    MThresh=float(options.MThresh)
    PAThresh=float(options.PAThresh)
    PCAVar=float(options.PCAVar)
    # runID = int(sys.argv[1])
    # numStreams = int(sys.argv[2])
    # socketOffset = int(sys.argv[3])
    # weightType = str(sys.argv[4])
    # CThresh = float(sys.argv[5])
    # MThresh = float(sys.argv[6])
    # MAX_WINDOW = int(sys.argv[7])#80
    INIT_DAYS = MAX_WINDOW#80
    STABLE_SIZE = 2* MAX_WINDOW
    # MODEL_HIST_THRESHOLD_PROB = 0.4
    THRESHOLD = MODEL_HIST_THRESHOLD_ACC# = THRESHOLD#0.5
    # DEFAULT_PRED = str(sys.argv[9])
    # CD_TYPE = str(sys.argv[10])
    # if CD_TYPE == 'RePro':
        # THRESHOLD = float(sys.argv[8])#0.5
        # MODEL_HIST_THRESHOLD_PROB = 0.4
        # MODEL_HIST_THRESHOLD_ACC = THRESHOLD#0.5
    # else:
        # ADWIN_DELTA = float(sys.argv[8])
        # MODEL_HIST_THRESHOLD_PROB = 0.4
        # MODEL_HIST_THRESHOLD_ACC = 0.6#0.5

    FPdict = dict()
    
    if DEFAULT_PRED == 'Following':
        if numStreams == 0:
            FPdict = get7FollowingFPdict()
            # numStreams = 7
        else:
            FPdict = getFollowingFPdict()
        # FPdict = getFollowingFPdict()
    elif DEFAULT_PRED == 'Heating':
        FPdict = getHeatingDates()
    else:
        FPdict = getFPdict(DEFAULT_PRED)
    journeyList = list(FPdict.keys())

    random.shuffle(journeyList)
    if numStreams == 0:
        journeyList = journeyList[0:7]
    else:
        journeyList = journeyList[0:numStreams]

    for idx, i in enumerate(journeyList):
        source = dict()
        source['Name'] = "source"+str(idx)+":"+str(i)
        source['uid'] = str(i)
        source['stdo']="TestResultsLog/Run"+str(runID)+"/"+str(weightType)+str(source['Name'])+str(numStreams)+"Out.txt"
        source['PORT'] = socketOffset+idx
        source['Run'] = "source"+str(CD_TYPE)+".py"
        # source['stdin'] = FPdict[i]
        source['weightType'] = weightType
        source['cullThresh'] = CThresh
        source['miThresh'] = MThresh
        source['paThresh']= PAThresh
        source['default_pred'] = DEFAULT_PRED
        source['PCAVar'] = PCAVar
        if DEFAULT_PRED == 'Heating':
            source['sFrom'] = FPdict[idx]['start']#startDates[i]#"2014-01-01"
            source['sTo'] = FPdict[idx]['end']#endDates[i]#"2014-02-28"
            source['stdin'] = "../HeatingSimDG/HeatingSimData/userSOURCEDataSimulation.csv"
        else:
            source['sFrom'] = 0
            source['sTo'] = 15000
            source['stdin'] = FPdict[i]

        sourceModels = dict()
        MODELS[idx] = sourceModels
        receivedModels = []
        sourceInfo[idx] = source

    print("creating threads")
    totalTime = 0
    for k,v in sourceInfo.items():
        print(k, v)
        print("making thread")
        sThread = myThread(k,v,receivedModels,runID,numStreams)
        print("starting thread")
        sThread.start()
        totalTime = 0
        while not MODELS[k]:
            tts = random.uniform(0.0,1.2)
            time.sleep(10)
            totalTime = totalTime+ 100
            if totalTime >= 300:
                print(" no stable models in :" +str(k))
                break
        print(" RECIEVED FIRST MODEL SO STARTING NEXT THREAD")
    

if __name__ == '__main__':main()

