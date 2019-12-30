import heapq # for retrieval topK
import numpy as np
import threading
import math

def saveFile(string, fileName):
    fp_w = open(fileName,'w')
    fp_w.write(string)
    fp_w.close()
    return 0

def loadGroundTruth(testFile):
    fp = open(testFile)
    lines = fp.readlines()
    groundTruthDict = {}
    userRecllDict = {}
    for line in lines:
        lineStr = line.strip().split('\t')
        userId = int(lineStr[0])
        itemId = int(lineStr[1])
        if userId not in groundTruthDict:
            groundTruthDict[userId] = {}
            groundTruthDict[userId][itemId] = 1
        groundTruthDict[userId][itemId] = 1
        if userId not in userRecllDict:
            userRecllDict[userId] = 1
        else:
            userRecllDict[userId] += 1
    return groundTruthDict, userRecllDict

def loadTrainDict(trainFile):
    fp = open(trainFile)
    lines = fp.readlines()
    trainDict = {}
    for line in lines:
        lineStr = line.strip().split('\t')
        userId = int(lineStr[0])
        itemId = int(lineStr[1])
        if userId not in trainDict:
            trainDict[userId] = {}
            trainDict[userId][itemId] = 1
        trainDict[userId][itemId] = 1
    return trainDict

def precisionAndRecall(sorted_x, groundTruthDict, userRecallNum, k):
    hitCount = 0
    recallByUser = 0.0
    #hitList = []
    sorted_x = sorted_x[:k]
    for itemIndex in sorted_x:
        #pairId = str(userIndex) + '-' + str(itemIndex)
        if itemIndex in groundTruthDict:
            hitCount += 1
    #hitList.append(pairId)

    dcgPerUser = 0
    idcgPerUser = 0
    rank = 0
    if userRecallNum!=0:
        groundtruthLength = userRecallNum#userRecallDict[userIndex]
        if groundtruthLength >= k:
            groundtruthLength = k
        for index in range(groundtruthLength):
            rank = rank + 1
            idcgPerUser += 1 / float(math.log(rank + 1, 2))
        
        rank = 0
        for itemIndex in sorted_x:
            #pairId = str(userIndex) + '-' + str(itemIndex)
            if itemIndex in groundTruthDict:
                position = rank + 1
                dcgPerUser += 1 / float(math.log(position + 1, 2))
            rank = rank + 1
        
        ndcgPerUser = float(dcgPerUser) / float(idcgPerUser)
    # if idcgPerUser!=0:
    #     ndcgPerUser = float(dcgPerUser) / float(idcgPerUser)
    # else:
    #     ndcgPerUser = 0.0
    else:
        ndcgPerUser = 0.0

    if userRecallNum != 0:
        recallByUser = hitCount * 1.0 / userRecallNum
        # print userIndex, k, hitCount, hitList
        
    return hitCount * 1.0 / k, recallByUser, ndcgPerUser

def evaluateModel(model, dataUser, dataCollection, K, dataSetName, epoch, list_num, item_num):
    positionVector = np.arange(0, item_num, step=1)
    print 'number of position: ', len(positionVector)
    positionMatrix = []
    for i in range(list_num):
        positionMatrix.append(positionVector)
    
    trainFile = ''
    testFile = ''
    if dataSetName=='goodreads':
        trainFile = '/home/infolab/atlist/data/goodreads/train.txt'
        testFile = '/home/infolab/atlist/data/goodreads/test.txt'
    elif dataSetName== 'spotify':
        trainFile = '/home/infolab/atlist/data/spotify/train.txt'
        testFile = '/home/infolab/atlist/data/spotify/test.txt'
    elif dataSetName=='zhihuLarge':
        trainFile = '/home/infolab/atlist/data/zhihuLarge/train.txt'
        testFile = '/home/infolab/atlist/data/zhihuLarge/test.txt'

    trainDict = loadTrainDict(trainFile)
    groundTruthDict, userRecllDict = loadGroundTruth(testFile)
    #print dataUser
    print 'what!'
    #userNum = dataUser.shape[0]
    userNum = dataUser.shape[0]
    itemNum = dataCollection.shape[0]

    print 'number of user: ', userNum
    print 'number of item: ', itemNum
    totalprecision50 = 0
    totalrecall50 = 0
    totalprecision100 = 0
    totalrecall100 = 0
    totalprecision20 = 0
    totalrecall20 = 0
    totalprecision10 = 0
    totalrecall10 = 0
    totalprecision5 = 0
    totalrecall5 = 0
    totalprecision1 = 0
    totalrecall1 = 0
    
    totalNDCG50 = 0
    totalNDCG100 = 0
    totalNDCG20 = 0
    totalNDCG10 = 0
    totalNDCG5 = 0
    totalNDCG1 = 0
    
    count = 0
    #userNum = 10
    for userIndex in range(userNum):
        count+=1
        if count%100==0:
            print count
        userTensor = dataUser[userIndex]
        #user_input = np.full(itemNum, userTensor)
        user_input = []
        user_oneHot_input = []
        user_position_input = []
        item_position_input = []
            #user_position_input = []
            #item_position_input = []
        for i in range(itemNum):
            user_input.append(userTensor)
            user_oneHot_input.append(userIndex)
            user_position_input.append(positionMatrix)
            item_position_input.append(positionVector)

        user_oneHot_input = np.array(user_oneHot_input)
        user_input = np.array(user_input)
        item_input = dataCollection
        item_oneHot_input = np.arange(itemNum)
        user_position_input = np.array(user_position_input)
        item_position_input = np.array(item_position_input)
        predictions = model.predict([user_input, item_input, user_oneHot_input, item_oneHot_input, user_position_input, item_position_input], batch_size=2048, verbose=0)
        #print 'done with one user'
        #print predictions[:100]

        userTrainDict = trainDict[userIndex]
        map_item_score = {}

        for itemIndex in xrange(itemNum):
            if itemIndex not in userTrainDict:
                map_item_score[itemIndex] = predictions[itemIndex]
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        #print ranklist

        if userIndex in groundTruthDict:
            groundTruthDictUser = groundTruthDict[userIndex]
        else:
            groundTruthDictUser = {}
        if userIndex in userRecllDict:
            userRecallNum = userRecllDict[userIndex]
        else:
            userRecallNum = 0
        precision1, recall1, ndcgPerUser1 = precisionAndRecall(ranklist, groundTruthDictUser, userRecallNum, 1)
        precision5, recall5, ndcgPerUser5 = precisionAndRecall(ranklist, groundTruthDictUser, userRecallNum, 5)
        precision10, recall10, ndcgPerUser10 = precisionAndRecall(ranklist, groundTruthDictUser, userRecallNum, 10)
        precision20, recall20, ndcgPerUser20 = precisionAndRecall(ranklist, groundTruthDictUser, userRecallNum, 20)
        precision50, recall50, ndcgPerUser50 = precisionAndRecall(ranklist, groundTruthDictUser, userRecallNum, 50)
        precision100, recall100, ndcgPerUser100 = precisionAndRecall(ranklist, groundTruthDictUser, userRecallNum, 100)
        #print precison, recallByUser
        totalprecision50 += precision50
        totalrecall50 += recall50
        totalprecision100 += precision100
        totalrecall100 += recall100
        totalprecision20 += precision20
        totalrecall20 += recall20
        totalprecision10 += precision10
        totalrecall10 += recall10
        totalprecision5 += precision5
        totalrecall5 += recall5
        totalprecision1 += precision1
        totalrecall1 += recall1

        totalNDCG50 += ndcgPerUser50
        totalNDCG100 += ndcgPerUser100
        totalNDCG20 += ndcgPerUser20
        totalNDCG10 += ndcgPerUser10
        totalNDCG5 += ndcgPerUser5
        totalNDCG1 += ndcgPerUser1

    txt = ''
    print 'precision@1:', totalprecision1 / userNum
    print 'recall@1:', totalrecall1 / userNum
    print 'precision@5:', totalprecision5 / userNum
    print 'recall@5:', totalrecall5 / userNum
    print 'precision@10:', totalprecision10 / userNum
    print 'recall@10:', totalrecall10 / userNum
    print 'precision@20:', totalprecision20 / userNum
    print 'recall@20:', totalrecall20 / userNum
    print 'precision@50:', totalprecision50 / userNum
    print 'recall@50:', totalrecall50 / userNum
    print 'precision@100:', totalprecision100 / userNum
    print 'recall@100:', totalrecall100 / userNum
    print 'NDCG@1:', totalNDCG1 / userNum
    print 'NDCG@5:', totalNDCG5 / userNum
    print 'NDCG@10:', totalNDCG10 / userNum
    print 'NDCG@20:', totalNDCG20 / userNum
    print 'NDCG@50:', totalNDCG50 / userNum
    print 'NDCG@100:', totalNDCG100 / userNum
    
    txt += 'precision@1: ' + str(totalprecision1 / userNum)+ '\n'
    txt += 'recall@1: ' + str(totalrecall1 / userNum)+ '\n'
    txt += 'precision@5: ' + str(totalprecision5 / userNum)+ '\n'
    txt += 'recall@5: ' + str(totalrecall5 / userNum)+ '\n'
    txt += 'precision@10: ' + str(totalprecision10 / userNum)+ '\n'
    txt += 'recall@10: ' + str(totalrecall10 / userNum)+ '\n'
    txt += 'precision@20: ' + str(totalprecision20 / userNum)+ '\n'
    txt += 'recall@20: ' + str(totalrecall20 / userNum)+ '\n'
    txt += 'precision@50: ' + str(totalprecision50 / userNum)+ '\n'
    txt += 'recall@50: ' + str(totalrecall50 / userNum)+ '\n'
    txt += 'precision@100: ' + str(totalprecision100 / userNum)+ '\n'
    txt += 'recall@100: ' + str(totalrecall100 / userNum)+ '\n'
    txt += 'NDCG@1: ' + str(totalNDCG1 / userNum)+ '\n'
    txt += 'NDCG@5: ' + str(totalNDCG5 / userNum)+ '\n'
    txt += 'NDCG@10: ' + str(totalNDCG10 / userNum)+ '\n'
    txt += 'NDCG@20: ' + str(totalNDCG20 / userNum)+ '\n'
    txt += 'NDCG@50: ' + str(totalNDCG50 / userNum)+ '\n'
    txt += 'NDCG@100: ' + str(totalNDCG100 / userNum)+ '\n'
    filename = dataSetName + str(epoch) + '.res'
    saveFile(txt, filename)
    return
