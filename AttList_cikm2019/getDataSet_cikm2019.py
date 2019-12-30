import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from keras.layers import Embedding
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import json

def getEmbeddingLayer(itemNum, EMBEDDING_DIM, MAX_ITEM_NUM):
    embedding_layer = Embedding(itemNum + 1,
                                EMBEDDING_DIM,
                                name = 'item_embedding_layer',
                                embeddings_regularizer=l2(0.000001),#
                                #weights=[embedding_matrix],
                                input_length=MAX_ITEM_NUM,
                                trainable=True,
                                )#mask_zero=True
    return embedding_layer

def getUserEmbeddingLayer(userNum, EMBEDDING_DIM):
    MF_Embedding_User = Embedding(input_dim = userNum, output_dim = EMBEDDING_DIM, name = 'user_oneHot_embedding',
                                  W_regularizer = l2(0.000001), input_length=1)
    
    return MF_Embedding_User

def getListEmbeddingLayer(listNum, EMBEDDING_DIM):
    MF_Embedding_Collection = Embedding(input_dim = listNum, output_dim = EMBEDDING_DIM, name = 'collection_oneHot_embedding',
                                   W_regularizer = l2(0.000001), input_length=1)
    return MF_Embedding_Collection

def getPositionEmbeddingLayer(MAX_ITEM_NUM, EMBEDDING_DIM):
    itemPosEmbedding = Embedding(input_dim = MAX_ITEM_NUM, output_dim = EMBEDDING_DIM, name='user_item_position_emb_layer',
                                 embeddings_regularizer=l2(0), input_length=MAX_ITEM_NUM)
    return itemPosEmbedding

def getUserCollectionData(trainDataFile, listItemFile, MAX_ITEM_NUM, MAX_LIST_NUM):
    fp = open(trainDataFile)
    lines = fp.readlines()
    userListTrainDict = {}
    for line in lines:
        lineStr = line.strip().split('\t')
        userId = int(lineStr[0])
        listId = int(lineStr[1])
        if userId not in userListTrainDict:
            userListTrainDict[userId] = [listId]
        else:
            userListTrainDict[userId].append(listId)
    fp.close()

    fp = open(listItemFile)
    lines = fp.readlines()
    listItemDict = {}
    itemUniqueDict = {}
    for line in lines:
        lineStr = line.strip().split('\t')
        listId = int(lineStr[0])
        itemList = json.loads(lineStr[1])
        listItemDict[listId] = itemList
        for itemId in itemList:
            itemUniqueDict[itemId] = itemUniqueDict.get(itemId, 0) + 1
    fp.close()

    userNum = len(userListTrainDict)
    listNum = len(listItemDict)
    itemNum = len(itemUniqueDict)
    print 'number of users: ', userNum
    print 'number of lists: ', listNum
    print 'number of items: ', itemNum

    dataUser = np.zeros((len(userListTrainDict), MAX_LIST_NUM, MAX_ITEM_NUM), dtype='int32')
    dataCollection = np.zeros((len(listItemDict), MAX_ITEM_NUM), dtype='int32')

    for userId in userListTrainDict:
        listsFollowedByUser = userListTrainDict[userId]
        minListNum = min(MAX_LIST_NUM, len(listsFollowedByUser))
        for j in range(minListNum):
            listId = listsFollowedByUser[j]
            itemsInList = listItemDict[listId]
            length = min(MAX_ITEM_NUM, len(itemsInList))
            for i in range(length):
                dataUser[userId, j, i] = itemsInList[i]

    for listId in listItemDict:
        itemsInList = listItemDict[listId]
        length = min(MAX_ITEM_NUM, len(itemsInList))
        for i in range(length):
            dataCollection[listId, i] = itemsInList[i]

    print('Shape of user data tensor:', dataUser.shape)
    print('Shape of collection data tensor:', dataCollection.shape)
    return dataUser, dataCollection, userNum, listNum, itemNum

def getOriginalTrainData(trainDataFile):
    fp = open(trainDataFile)
    lines = fp.readlines()
    positiveTrainList = []
    for line in lines:
        lineStr = line.strip().split('\t')
        userId = int(lineStr[0])
        itemId = int(lineStr[1])
        positiveTrainList.append([userId, itemId, 1])
    return positiveTrainList

def pickUpDataset(dataSetName, item_embedding_size, max_item_num, max_list_num):
    if dataSetName=='goodreads':
        dataUser, dataCollection, userNum, listNum, itemNum = getUserCollectionData('./data/goodreads/train.txt', 
            './data/goodreads/listItem_goodreads.txt', max_item_num, max_list_num)
        positiveTrainList = getOriginalTrainData(
            './data/goodreads/train.txt')
        embedding_layer = getEmbeddingLayer(itemNum, item_embedding_size, max_item_num)
        MF_Embedding_User = getUserEmbeddingLayer(userNum, item_embedding_size)
        MF_Embedding_Collection = getListEmbeddingLayer(listNum, item_embedding_size)
        itemPosEmbedding = getPositionEmbeddingLayer(max_item_num, item_embedding_size)


    elif dataSetName=='spotify':
        dataUser, dataCollection, userNum, listNum, itemNum = getUserCollectionData('./data/spotify/train.txt', 
            './data/spotify/listItem_spotify.txt', max_item_num, max_list_num)
        positiveTrainList = getOriginalTrainData('./data/spotify/train.txt')
        embedding_layer = getEmbeddingLayer(itemNum, item_embedding_size, max_item_num)
        MF_Embedding_User = getUserEmbeddingLayer(userNum, item_embedding_size)
        MF_Embedding_Collection = getListEmbeddingLayer(listNum, item_embedding_size)
        itemPosEmbedding = getPositionEmbeddingLayer(max_item_num, item_embedding_size)
        

    elif dataSetName=='zhihu':
        dataUser, dataCollection, userNum, listNum, itemNum = getUserCollectionData('./data/zhihu/train.txt', 
            './data/zhihu/listItem_zhihu.txt', max_item_num, max_list_num)
        positiveTrainList = getOriginalTrainData('./data/zhihu/train.txt')
        embedding_layer = getEmbeddingLayer(itemNum, item_embedding_size, max_item_num)
        MF_Embedding_User = getUserEmbeddingLayer(userNum, item_embedding_size)
        MF_Embedding_Collection = getListEmbeddingLayer(listNum, item_embedding_size)
        itemPosEmbedding = getPositionEmbeddingLayer(max_item_num, item_embedding_size)

    return dataUser, dataCollection, positiveTrainList, embedding_layer, MF_Embedding_User, MF_Embedding_Collection, itemPosEmbedding