import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from . import residual_net


def create_network(images, num_classes=None, reuse=None,
                   create_summaries=True, weight_decay=1e-8):
    '''
    創建神經網路函數
    Parameters
    ----------
    images [height,width,3]:
        輸入圖片
    num_classes int:
        行人類別數量
    reuse True or None:
        conv2d之變數
    create_summaries bool:
        是否畫在TensorBoard上
    Returns
    -------
    features List[batch size,128]:
        經過l2 normalize之後的輸出特徵
    logits List[batch size,num_classes]:
        (每個人的類別的為歸一化的概率)找出屬於哪一個行人類別之特徵
    '''
    #設定activation function
    nonlinearity = tf.nn.elu
    #建議先初始化權重
    conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)     
    #初始化bias
    conv_bias_init = tf.zeros_initializer()
    #宣告正則化函數
    conv_regularizer = slim.l2_regularizer(weight_decay)
    #建議先初始化權重
    fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    #初始化bias
    fc_bias_init = tf.zeros_initializer()
    #宣告正則化函數
    fc_regularizer = slim.l2_regularizer(weight_decay)

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    network = images                                                            #batch size*height*width*channel      128*128*64*3
    network = slim.conv2d(                                                      #output:128*128*64*32    3*3*3*32
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_1",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    lay=network[:,:,:,0:3]                                                      #第一維只能選1、3、4個維度
    #繪製conv1_1的特徵圖
    if create_summaries:
        #tf.summary.histogram(network.name + "/activations", network)
        tf.summary.image("conv1_1/feature_map", lay,                                #32個特徵圖、RGB3維
                         max_outputs=128)
    network = slim.conv2d(                                                      #128*128*64*32      3*3*32*32
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_2",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)

    network = slim.max_pool2d(                                                  #128*64*32*32
        network, [3, 3], [2, 2], scope="pool1", padding="SAME")

    network = residual_net.residual_block(                                      #128*64*32*32
        network, "conv2_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False, is_first=True,
        summarize_activations=create_summaries)
    network = residual_net.residual_block(                                      #128*64*32*32
        network, "conv2_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)
    network = residual_net.residual_block(                                      #128*64*32*32
        network, "conv2_5", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_net.residual_block(                                      #128*32*16*64
        network, "conv3_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,                                    #增加維度
        summarize_activations=create_summaries)
    print(network.get_shape().as_list(),' networkkkkkkkkkkk')
    network = residual_net.residual_block(                                      #128*32*16*64
        network, "conv3_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)
    network = residual_net.residual_block(                                      #128*32*16*64
        network, "conv3_5", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_net.residual_block(                                      #128*16*8*128
        network, "conv4_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,                                    #增加維度
        summarize_activations=create_summaries)
    print(network.get_shape().as_list(),' networkkkkkkkkkkk')

    network = residual_net.residual_block(                                      #128*16*8*128
        network, "conv4_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)
    network = residual_net.residual_block(                                      #128*16*8*128
        network, "conv4_5", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    
    feature_dim = network.get_shape().as_list()[-1]
    print("feature dimensionality: ", feature_dim)
    #network = slim.flatten(network)                         #128*16384
    network=tf.keras.layers.GlobalAveragePooling2D()(network)               #128*128
    
    network = slim.dropout(network, keep_prob=0.6)

    network = slim.fully_connected(                         #128*128
        network, feature_dim, activation_fn=nonlinearity,
        normalizer_fn=batch_norm_fn, weights_regularizer=fc_regularizer,
        scope="fc1", weights_initializer=fc_weight_init,
        biases_initializer=fc_bias_init)

    features = network

    # Features in rows, normalize axis 1.
    features = tf.nn.l2_normalize(features, dim=1)              #128*128每個人的特徵正規化
    #計算參數量
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('parameter=',total_parameters)

    with slim.variable_scope.variable_scope("ball", reuse=reuse):
        weights = slim.model_variable(                              #生成截断正态分布的随机数   128*2010
            "mean_vectors", (feature_dim, int(num_classes)),
            initializer=tf.truncated_normal_initializer(stddev=1e-3),
            regularizer=None)
        scale = slim.model_variable(                                #模型參數，會隨著訓練被調整
            "scale", (), tf.float32,
            initializer=tf.constant_initializer(0., tf.float32),
            regularizer=slim.l2_regularizer(1e-1))
        if create_summaries:
            tf.summary.scalar("scale", scale)
        scale = tf.nn.softplus(scale)                               #log(exp(scale),+1)

        # Mean vectors in colums, normalize axis 0.
        weights_normed = tf.nn.l2_normalize(weights, dim=0)             #權重正規化
        logits = scale * tf.matmul(features, weights_normed)            #128*2010
    return features, logits                                             

def create_network_fn(is_training, num_classes, 
                           weight_decay=1e-8, reuse=None):
    def factory_fn(image):
        '''
        使用slim以簡潔code建構網路
        只有前面[]內的修飾函數才能夠使用arg_scope
        '''
        with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.fully_connected,slim.batch_norm, slim.layer_norm],reuse=reuse):
                features, logits = create_network(
                    image, num_classes=num_classes, 
                    reuse=reuse, create_summaries=is_training,
                    weight_decay=weight_decay)
                return features, logits

    return factory_fn
