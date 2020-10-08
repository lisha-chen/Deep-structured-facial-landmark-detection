from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
# import tensorflow_probability as tfp
# NHWC

dim1 = 64
dim2 = 64

x = np.linspace(0, dim1-1, dim1, dtype=np.float32)
y = np.linspace(0, dim2-1, dim2, dtype=np.float32)
mesh = np.transpose(np.array(np.meshgrid(x, y)).reshape((2,-1)))

down_scale = 255./63.

epsilon = 1e-5
# tfd = tfp.distributions


#------------------------------------------------------------------------------

def softmax_nll_with_logits(logits, labels):
    int_labels = tf.clip_by_value(tf.floor(labels / down_scale),
                    clip_value_min=0,
                    clip_value_max=tf.cast(tf.shape(logits)[2], tf.float32)-1) 
                                # batch_size * # heatmaps * 2
    flat_labels = int_labels[:, :, 1] * \
        tf.cast(tf.shape(logits)[2], tf.float32) + int_labels[:, :, 0]
    # y * width + x
    # print(flat_labels.shape)
    output_flat = tf.transpose(tf.reshape(logits, 
                [-1, dim1 * dim2, logits.shape[-1]]),
                perm=[0, 2, 1]) 
                # (N C H*W)
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(flat_labels, tf.int32), logits=output_flat))
    return loss



def L1_mean_loss(mean, labels):
    outlabels = labels / down_scale
    
    loss = tf.reduce_mean(tf.keras.metrics.mae(outlabels, mean))
    
    return loss


def multi_gaussian_fullcov_nll(mean, inv_cov, logdet_invcov, labels):
    # with tf.device('/CPU:0'):
    outlabels = labels / down_scale
    
    y_diff = tf.expand_dims(tf.subtract(outlabels, mean), -1)
    
    loss = tf.reduce_mean((tf.matmul(tf.matmul(y_diff, inv_cov, transpose_a=True), y_diff)) \
                        - tf.expand_dims(tf.expand_dims(logdet_invcov,-1),-1)) 
    
    return loss


