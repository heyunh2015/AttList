from keras.engine.topology import Layer
from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import *
from keras.regularizers import l2

class AttLayerSelf(Layer):
    def __init__(self, W_regularizer):#, attention_dim, W_regularizer
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.W_regularizer = regularizers.get(W_regularizer)
        self.dropout = Dropout(0.5)
        super(AttLayerSelf, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        super(AttLayerSelf, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        q = x#K.dot(x, self.W_q)#K.bias_add(K.dot(x, self.W_q), self.b_q)
        v = x#K.dot(x, self.W_v)#K.bias_add(K.dot(x, self.W_v), self.b_v)
        k = x#K.dot(x, self.W_k)#K.bias_add(K.dot(x, self.W_k), self.b_k)
        att = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/10.0)([q, k])
        weight = Activation('softmax')(att)
        weight = self.dropout(weight)
        output = K.batch_dot(weight, v)
        output = output + x
        
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])
