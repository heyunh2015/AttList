# author - Yun He, Texas A&M University
# Dec 26 2019
import json
import time

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.topology import Layer
from keras.layers import Dense, Input, GlobalAveragePooling1D, Dropout
from keras.layers import Embedding, TimeDistributed, LSTM, GRU, Bidirectional, concatenate
from keras.layers import multiply, Add, Flatten, Lambda
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.models import Model
from keras.regularizers import l2
import argparse
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

from AttLayer_cikm2019 import AttLayer
from AttLayerSelf_cikm2019 import AttLayerSelf
import getDataSet_cikm2019
import evaluate_cikm2019

def parse_args():
    parser = argparse.ArgumentParser(description="atlist")
    parser.add_argument('--dataSetName', nargs='?', default='spotify',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100, #epochs that generate the results in the paper: goodreads 21, spotify 51, zhihu 42
                        help='Number of epochs.')
    parser.add_argument('--negativeNum', type=int, default=5, #recommended: goodreads 3, spotify 5, zhihu 7
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, #recommended: spotify zhihu 0.0001, goodreads 0.001
                        help='learning_rate.')
    parser.add_argument('--item_embedding_size', type=int, default=100, #recommended: zhihu 64, spotify goodreads 100
                        help='Embedding size of item.')
    parser.add_argument('--max_item_num', type=int, default=32,
                        help='For each list, maximum number of items to be considered in the model.')
    parser.add_argument('--max_list_num', type=int, default=15,  #
                        help='For each user, maximum number of lists to be considered in the model.')
    parser.add_argument('--item_embedding_dropout', type=float, default=0.3,
                        help='item_embedding_dropout.')
    parser.add_argument('--att_dropout', type=float, default=0.3,
                        help='att_dropout.')
    parser.add_argument('--valnilla_att_l2', type=float, default=0,#recommended: goodreads 1.0, spotify, zhihu 0
                        help='valnilla_att_l2.')
    parser.add_argument('--self_att_l2', type=float, default=0.001,#recommended: goodreads, spotify 0.001 zhihu 0
                        help='self_att_l2.')
    return parser

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

def getModel(embedding_layer, MF_Embedding_User, MF_Embedding_Collection, itemPosEmbedding, item_embedding_dropout, att_dropout, valnilla_att_l2, self_att_l2, MAX_LIST_NUM, MAX_ITEM_NUM):
    #part 1: User Representation Learning

    #input layer
    following_list_input = Input(shape=(2 * MAX_ITEM_NUM,), dtype='int32')
    following_list_input_items = crop(1, 0, MAX_ITEM_NUM)(following_list_input)
    following_list_input_position = crop(1, MAX_ITEM_NUM, MAX_ITEM_NUM * 2)(following_list_input)

    #Positional Item Representation Layer
    embedded_following_list_items = embedding_layer(following_list_input_items)
    embedded_following_list_items = Dropout(item_embedding_dropout)(embedded_following_list_items)
    embedded_following_list_position = itemPosEmbedding(following_list_input_position)
    embedded_following_list_position = Dropout(0.8)(embedded_following_list_position)
    embedded_following_list = Add()([embedded_following_list_items, embedded_following_list_position])

    #Item-level Self-Attentive Aggregation Layer
    embedded_following_list_self = AttLayerSelf(l2(self_att_l2))(embedded_following_list)
    following_list_aggregated = AttLayer(100, l2(valnilla_att_l2))(embedded_following_list_self)
    following_list_aggregated = Dropout(att_dropout)(following_list_aggregated)
    listEncoder = Model(following_list_input, following_list_aggregated)

    #input layer
    user_input = Input(shape=(MAX_LIST_NUM, MAX_ITEM_NUM,), dtype='int32', )#shape=(MAX_LIST_NUM, MAX_ITEM_NUM)
    user_position_input = Input(shape=(MAX_LIST_NUM, MAX_ITEM_NUM,), dtype='int32', )
    user_all_input = concatenate([user_input, user_position_input], axis=-1)

    #List-level Self-Attentive Aggregation Layer
    user_encoder = TimeDistributed(listEncoder, input_shape=())(user_all_input)
    user_encoder_self = AttLayerSelf(l2(self_att_l2))(user_encoder)
    user_aggregated = AttLayer(100, l2(valnilla_att_l2))(user_encoder_self)
    user_aggregated = Dropout(att_dropout)(user_aggregated)

    #part 2: List Representation Learning
    # input layer
    predict_list_input_items = Input(shape=(MAX_ITEM_NUM,), dtype='int32', )#shape=(MAX_ITEM_NUM,)
    predict_list_input_position = Input(shape=(MAX_ITEM_NUM,), dtype='int32', )

    #Positional Item Representation Layer
    embedded_predict_list_items = embedding_layer(predict_list_input_items)
    embedded_predict_list_items = Dropout(item_embedding_dropout)(embedded_predict_list_items)
    embedded_predict_list_position = itemPosEmbedding(predict_list_input_position)
    embedded_predict_list_position = Dropout(0.8)(embedded_predict_list_position)
    embedded_predict_list = Add()([embedded_predict_list_items, embedded_predict_list_position])
    
    #Item-level Self-Attentive Aggregation Layer
    embedded_predict_list_self = AttLayerSelf(l2(self_att_l2))(embedded_predict_list)
    predict_list_aggregated = AttLayer(100, l2(valnilla_att_l2))(embedded_predict_list_self)
    predict_list_aggregated = Dropout(att_dropout)(predict_list_aggregated)

    # Combining Layer
    user_oneHot_input = Input(shape=(1,), dtype='int32', name = 'user_oneHot_input')
    list_oneHot_input = Input(shape=(1,), dtype='int32', name = 'list_oneHot_input')

    user_single = Flatten()(Dropout(0.3)(MF_Embedding_User(user_oneHot_input)))
    list_single = Flatten()(Dropout(0.3)(MF_Embedding_Collection(list_oneHot_input)))
    
    l_att_user_combine = Add()([user_aggregated, user_single])#
    l_att_list_combine = Add()([predict_list_aggregated, list_single])#

    # Recommending Top-K Item Lists
    mf_vector = multiply([l_att_user_combine, l_att_list_combine])
    mlp_vector = concatenate([l_att_user_combine, l_att_list_combine])
    predict_vector = concatenate([mf_vector, mlp_vector])
    predict_vector = Dense(100, activation='relu')(predict_vector)
    predict_vector = Dropout(0.5)(predict_vector)
    preds = Dense(1, activation='sigmoid')(predict_vector)
    model = Model(inputs=[user_input, predict_list_input_items, user_oneHot_input, list_oneHot_input, user_position_input, predict_list_input_position], outputs=preds)
    return model

def getModelForPrintAttention(embedding_layer, MAX_ITEM_NUM):
    from AttLayerPrint import AttLayerPrint
    collection_input = Input(shape=(MAX_ITEM_NUM,), dtype='int32', )#shape=(MAX_ITEM_NUM,)
    embedded_collection = embedding_layer(collection_input)
    attention_scores = AttLayerPrint(100)(embedded_collection)
    model = Model(inputs=[collection_input], outputs=attention_scores)
    return model

def get_train_instance(positiveTrainList, neg=3):
    trainDict = {}
    uDimensionMax = 0
    iDimensionMax = 0
    for line in positiveTrainList:
        userId = line[0]
        itemId = line[1]
        uDimensionMax = max(userId, uDimensionMax)
        iDimensionMax = max(itemId, iDimensionMax)
        if str(userId)+'\t'+str(itemId) not in trainDict:
            trainDict[str(userId)+'\t'+str(itemId)] = 1
    positiveSampleNum = len(trainDict)
    count = 0
    print int(positiveSampleNum*neg)
    negativeTrainList = []
    while count<=int(positiveSampleNum*neg):
        userId = np.random.randint(0, uDimensionMax+1)
        itemId = np.random.randint(0, iDimensionMax+1)
        sampleId = str(userId)+'\t'+str(itemId)
        if sampleId not in trainDict:
            negativeTrainList.append([userId, itemId, 0])
            count += 1
    print count
    trainList = []
    trainList.extend(negativeTrainList)
    trainList.extend(positiveTrainList)
#positiveTrainList.extend(negativeTrainList)
    print 'number of total training samples', len(trainList)
    return trainList


def getTrainData(trainList, dataUser, dataCollection, max_list_num, max_item_num):
    positionVector = np.arange(0, max_item_num, step=1)
    print 'number of position: ', len(positionVector)
    positionMatrix = []
    for i in range(max_list_num):
        positionMatrix.append(positionVector)
    
    labels = []
    user_input = []
    item_input = []
    user_oneHot_input = []
    item_oneHot_input = []
    uer_position_input = []
    item_position_input = []
    for line in trainList:
        #lineStr = line.strip().split('\t')
        userId = line[0]
        itemId = line[1]
        user_input.append(dataUser[userId])
        item_input.append(dataCollection[itemId])
        user_oneHot_input.append(userId)
        item_oneHot_input.append(itemId)
        uer_position_input.append(positionMatrix)
        item_position_input.append(positionVector)
        
        label = line[2]
        labels.append(label)

    #labels = to_categorical(np.asarray(labelList))
    user_input = np.array(user_input)
    item_input = np.array(item_input)
    user_oneHot_input = np.array(user_oneHot_input)
    item_oneHot_input = np.array(item_oneHot_input)
    uer_position_input = np.array(uer_position_input)
    item_position_input = np.array(item_position_input)
    labels = np.array(labels)
    print user_input.shape
    print item_input.shape
    print labels.shape
    return user_input, item_input, labels, user_oneHot_input, item_oneHot_input, uer_position_input, item_position_input

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    dataSetName = args.dataSetName
    epochs = args.epochs
    negativeNum = args.negativeNum
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    item_embedding_size = args.item_embedding_size
    max_item_num = args.max_item_num
    max_list_num = args.max_list_num
    item_embedding_dropout = args.item_embedding_dropout
    att_dropout = args.att_dropout
    valnilla_att_l2 = args.valnilla_att_l2
    self_att_l2 = args.self_att_l2

    
    start = time.time()
    dataUser, dataCollection, positiveTrainList, embedding_layer, MF_Embedding_User, MF_Embedding_Collection, itemPosEmbedding = getDataSet_cikm2019.pickUpDataset(dataSetName, item_embedding_size, max_item_num, max_list_num)
    
    
    print 'Build model'
    model = getModel(embedding_layer, MF_Embedding_User, MF_Embedding_Collection, itemPosEmbedding, item_embedding_dropout, att_dropout, valnilla_att_l2, self_att_l2, max_list_num, max_item_num)
    model.compile(loss='binary_crossentropy',
                  #optimizer='adam',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['acc'])#
    model.summary()
    print 'compile successful'

    for epoch in range(1, epochs+1):
        print 'iter: ', str(epoch)
        trainList = get_train_instance(positiveTrainList, negativeNum)
        user_input, collection_input, labels, user_oneHot_input, item_oneHot_input, user_position_input, collection_position_input = getTrainData(trainList, dataUser, dataCollection, max_list_num, max_item_num)
        model.fit([user_input, collection_input, user_oneHot_input, item_oneHot_input, user_position_input, collection_position_input], labels, #validation_data=([user_input, collection_input], labels),
                  nb_epoch=1, batch_size=batch_size, shuffle=True)#
        if epoch >= 45 and (epoch)%3==0:
            #model.save_weights(dataSetName + str(epoch) +'.h5')
            evaluate_cikm2019.evaluateModel(model, dataUser, dataCollection, 100,
                                   dataSetName, epoch, max_list_num, max_item_num)
    end = time.time()
    print 'running time: ', end - start