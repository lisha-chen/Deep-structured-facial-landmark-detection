# -*- coding: utf-8 -*-
"""
Created on Dec 29 15:58:03 2018

@author: lisha
"""
import numpy as np
import tensorflow as tf
#import tensorflow.keras
from tensorflow.python.keras import layers, models, activations

import scipy.io as sio
#from tensorflow.keras.layers import Layer
# from tensorflow.python.keras import backend as K
from metrics import metrics_FA_crf as metric

FLAGS = tf.app.flags.FLAGS

# channel order
# tensorflow NxHxWxC

dim1 = 64
dim2 = 64

x = np.linspace(0, dim1-1, dim1, dtype=np.float32)
y = np.linspace(0, dim2-1, dim2, dtype=np.float32)
mesh = np.transpose(np.array(np.meshgrid(x, y)).reshape((2,-1)))

mesh_w = mesh[:,0] #x
mesh_h = mesh[:,1] #y

kernel_h = 1.0


down_scale = 255./63.


epsilon = 1e-7


mat_content = sio.loadmat('./facemodel/DM68_wild34.mat')
DM = mat_content['DM68'][0,0]



def conv3x3(out_planes, strd=1, padding='same', bias=False, var_scope="conv3x3"):
    "3x3 convolution with padding"
    
    return layers.Conv2D(filters=out_planes, kernel_size=3,
                     strides=strd, padding=padding, use_bias=bias, name=var_scope+'_conv')


def downsample(x, out_planes, var_scope="downsample", is_train=True):
    
    residual = tf.layers.batch_normalization(name=var_scope+'_bn', trainable=True, inputs=x, training=is_train)
    residual = activations.relu(residual)
    residual = layers.Conv2D(filters=out_planes, kernel_size=1, strides=1, 
                                use_bias=False, name=var_scope+'_conv1')(residual)
    return residual


def ConvBlock(x, in_planes, out_planes, var_scope="ConvBlock", is_train=True):
    # bottleneck blocks
    if in_planes != out_planes:
        residual = downsample(x, out_planes=out_planes, var_scope=var_scope+'_downsample')
    else:
        residual = x

    out1 = tf.layers.batch_normalization(name=var_scope+'_bn1', trainable=True, inputs=x, training=is_train)           
    out1 = activations.relu(out1)
    out1 = conv3x3(out_planes=int(out_planes / 2), var_scope=var_scope+'_out1')(out1)
    
    out2 = tf.layers.batch_normalization(name=var_scope+'_bn2', trainable=True, inputs=out1, training=is_train)
    out2 = activations.relu(out2)
    out2 = conv3x3(out_planes=int(out_planes / 4), var_scope=var_scope+'_out2')(out2)
    
    out3 = tf.layers.batch_normalization(name=var_scope+'_bn3', trainable=True, inputs=out2, training=is_train)
    out3 = activations.relu(out3)
    out3 = conv3x3(out_planes=int(out_planes / 4), var_scope=var_scope+'_out3')(out3)

    out3 = layers.concatenate([out1, out2, out3], axis=-1, name=var_scope+'_concatenate')

    out3 += residual

    return out3


def mask_upper_tri_IJ(in_tensor):
    # in_tensor: ... C C

    ones = tf.ones([FLAGS.num_lmks, FLAGS.num_lmks])
    # print(ones.shape)
    mask_a = tf.linalg.band_part(ones, num_lower=0, num_upper=-1) # Upper triangular matrix
    mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool) # Make a bool mask
    # print(mask.shape)
    # ... m
    in_tensor_IJ = tf.boolean_mask(in_tensor, mask, axis=2)
    return in_tensor_IJ



class HourGlass():
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules # number of stacks
        self.depth = depth # number of downsample in one module
        self.features = num_features

        self._modules = {}
        self._generate_network(self.depth)
        
        self.intrinsic_feature = []

    def _generate_network(self, level, var_scope="HourGlass"):
        with tf.variable_scope(var_scope):  
            
            self._modules['b1_' + str(level)] = ConvBlock
            self._modules['b2_' + str(level)] = ConvBlock

            if level > 1:                    
                self._generate_network(level - 1, var_scope="HourGlass_level_%d"%level)
            else:                
                self._modules['b2_plus_' + str(level)] = ConvBlock
            
            self._modules['b3_' + str(level)] = ConvBlock

    def _forward(self, level, inp, var_scope, is_train):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1, 256, 256, 
                    var_scope=var_scope+"_ConvBlock_b1_level_%d"%level, is_train=is_train)

        # Lower branch
        low1 = layers.AveragePooling2D(pool_size=(2, 2), strides=2, 
                    name=var_scope+'_AvgPool2D_level_%d'%level)(inp)
        low1 = self._modules['b2_' + str(level)](low1, 256, 256, 
                    var_scope=var_scope+"_ConvBlock_b2_level_%d"%level, is_train=is_train)

        if level > 1:
            low2 = self._forward(level - 1, low1, 
                var_scope=var_scope+'_forward_level_%d'%level, is_train=is_train)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2, 256, 256, 
                var_scope=var_scope+"_ConvBlock_b2_plus_level_%d"%level, is_train=is_train)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3, 256, 256, 
                var_scope=var_scope+"_ConvBlock_b3_level_%d"%level, is_train=is_train)

        up2 = layers.UpSampling2D(size=(2, 2), name=var_scope+'_UpSample2D')(low3) 
        # interpolation='nearest', interpolation='bilinear'
        # up2 = nearest_upsampling(low3, 2)
        
        return up1 + up2

    def forward(self, x, var_scope='HG', is_train=True):
        return self._forward(self.depth, x, var_scope=var_scope, is_train=is_train)


class FAN():

    def __init__(self, num_modules=1):
        super(FAN, self).__init__()
        self.num_modules = num_modules

        # Base part
        with tf.variable_scope("FAN"):
            self.conv1 = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name='FAN_begin_conv1')
            self.bn1 = tf.layers.batch_normalization

            self._modules = {}
            # Stacking part
            for hg_module in range(self.num_modules):
                with tf.variable_scope("hg_module_%d"%hg_module):
                    
                    self._modules['m' + str(hg_module)] = HourGlass(1, 4, 256)
                    
                    self._modules['top_m_' + str(hg_module)] = ConvBlock
                    
                    self._modules['conv_last' + str(hg_module)] = layers.Conv2D(filters=256,
                                                            kernel_size=1, strides=1, padding='same', 
                                                            name='conv_last_HG_%d'%hg_module)
                    
                    self._modules['bn_end' + str(hg_module)] = tf.layers.batch_normalization
                    
                    self._modules['l' + str(hg_module)] = layers.Conv2D(filters=FLAGS.num_lmks, 
                                                            kernel_size=1, strides=1, padding='same', 
                                                            name='l_HG_%d'%hg_module)

                    if hg_module < self.num_modules - 1:
                        
                        self._modules['bl' + str(hg_module)] = layers.Conv2D(filters=256, 
                                                            kernel_size=1, strides=1, padding='same', 
                                                            name='bl_HG_%d'%hg_module)
                        self._modules['al' + str(hg_module)] = layers.Conv2D(filters=256, 
                                                            kernel_size=1, strides=1, padding='same', 
                                                            name='al_HG_%d'%hg_module)
        # inv part
        with tf.variable_scope("inv"):
            self.dense1 = layers.Dense(units=10, activation="relu", 
                            # kernel_initializer='zeros',
                            bias_initializer='zeros', name="inv_dense1")
            self.dense2 = layers.Dense(units=10,
                            # kernel_initializer='zeros',
                            bias_initializer='zeros', name="inv_dense2")
            self.dense3 = layers.Dense(units=3,
                            # kernel_initializer='zeros',
                            bias_initializer='zeros', name="inv_dense3")
            self.conv1_inv = layers.Conv2D(filters=256, kernel_size=3, strides=1, 
                                        padding='same', name='inv_conv1')
            self.conv2_inv = layers.Conv2D(filters=FLAGS.num_lmks, kernel_size=3, strides=1, 
                                        padding='same', name='inv_conv2')

        with tf.variable_scope("pairwise"):     
            # C(C+1)/2 3
            self.lij_ = tf.Variable(initial_value=0.01*tf.stack([tf.ones([FLAGS.num_lmks*(FLAGS.num_lmks+1)/2]),
                                        tf.zeros([FLAGS.num_lmks*(FLAGS.num_lmks+1)/2]),
                                        tf.ones([FLAGS.num_lmks*(FLAGS.num_lmks+1)/2])], axis=1), 
                            trainable=True,
                            name='cij/l_fac')

    def forward(self, x, is_train=True):
        # with tf.variable_scope("FAN"):
        x = activations.relu(self.bn1(self.conv1(x), training=is_train, name='FAN_begin_bn1', trainable=True))
        with tf.variable_scope("FAN_start"):
            x = layers.AveragePooling2D(pool_size=(2, 2), 
                    strides=2, name='AvgPool2D_layer1')(ConvBlock(x, 64, 128, 
                        var_scope="ConvBlock_layer1", is_train=is_train))
            
            x = ConvBlock(x, 128, 128, var_scope="ConvBlock_layer2", is_train=is_train)           
            x = ConvBlock(x, 128, 256, var_scope="ConvBlock_layer3", is_train=is_train)

        previous = x

        outputs = []
        self.last_feature = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)].forward(previous, var_scope='HG_%d'%i, is_train=is_train)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll, 256, 256, 
                        var_scope="ConvBlock_top_m_HG_%d"%i, is_train=is_train)

            ll = activations.relu(self._modules['bn_end' + str(i)](
                self._modules['conv_last' + str(i)](ll), training=is_train, 
                name='bn_end_HG_%d'%i, trainable=True))

            # Predict heatmaps
            # tmp_out, output of each stage
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)
            self.last_feature.append(ll)
            if i < self.num_modules - 1:
                # form previous, input to the next HG stage
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            self.logits = outputs
        return outputs

    def compute_mean(self, outputs):

        self.output_mean = []
        self.output_cov_wh = []
        for i in range(len(outputs)):
            output = outputs[i]
            # N HW C
            output_p = tf.nn.softmax(tf.reshape(output, 
                    [-1, tf.shape(output)[1] * tf.shape(output)[2], 
                    tf.shape(output)[-1]]), axis=1)

            # N HW C 2
            mesh_tile = tf.tile(tf.reshape(mesh, [1, mesh.shape[0], 1, mesh.shape[1]]), # 1 HW 1 2
                [tf.shape(output_p)[0], 1, FLAGS.num_lmks, 1]) # N HW C 2

            # N C 2
            output_mean = tf.reduce_sum(tf.tile(tf.expand_dims(output_p, -1), 
                [1,1,1,2]) * mesh_tile, axis=1)

            self.output_mean.append(output_mean)

        return self.output_mean

        
    def M_from_y(self, pts2d, C_IJ_, y3d_N, DM=DM):        
        # % pts2d: N C 2
        # % cij: 2 2 C C
        # % C_IJ_: 2m 2m
        # % y3d_N: N 3 C
        # y_IJ_post: N 2 m
        # y_IJ_3d_post: N 3 m

        # ----- % get y_IJ_post -----
        y = tf.transpose(pts2d, perm=[0,2,1]) # N 2 C            
        # N 2 C C # N 2 i j
        yij = tf.tile(tf.expand_dims(y, -1), [1,1,1,FLAGS.num_lmks]) - \
                tf.tile(tf.expand_dims(y, 2), [1,1,FLAGS.num_lmks,1])  # N 2 1 C
        # N 2 m
        y_IJ_post = tf.cast(mask_upper_tri_IJ(yij), dtype=tf.float32)

        # print(y_IJ_post.shape)
        #----- % get y_IJ_3d_post -----
        # N 3 C C # N 3 i j
        y_IJ_3d = tf.tile(tf.expand_dims(y3d_N, -1), [1,1,1,FLAGS.num_lmks]) - \
                tf.tile(tf.expand_dims(y3d_N, 2), [1,1,FLAGS.num_lmks,1])  # N 3 1 C
        # N 3 m
        y_IJ_3d_post = tf.cast(mask_upper_tri_IJ(y_IJ_3d), dtype=tf.float32)
        
        self.y_IJ_3d_post = y_IJ_3d_post
        self.y_IJ_post = y_IJ_post

        #----- % get F -----
        zeros_F = tf.zeros_like(y_IJ_3d_post)
        F = tf.concat([y_IJ_3d_post, zeros_F, zeros_F, y_IJ_3d_post], 1) # N 12 m
        F = tf.transpose(F, perm=[0,2,1]) # N m 12
        F = tf.reshape(F, [tf.shape(zeros_F)[0], 2*tf.shape(zeros_F)[2], 6]) # N 2m 6
        F = tf.cast(F, dtype=tf.float32)
        
        #----- % get M -----
        CF = tf.transpose(tf.tensordot(C_IJ_, F, axes=[[1], [1]]), perm=[1,0,2]) # N 2m 6
        Cy = tf.transpose(tf.tensordot(C_IJ_, 
            tf.reshape(tf.transpose(y_IJ_post, perm=[0,2,1]), [-1, 2*tf.shape(y_IJ_post)[-1]]), # N 2m
            axes=[[1], [1]]),
            perm=[1,0])  # N 2m
        self.CF = CF
        #M = (F'* C_IJ *F)\F'*C_IJ*y_IJ_post; %6*1
        mat = tf.matmul(F, CF, transpose_a=True) # N 6 6
        rhs = tf.matmul(F, tf.expand_dims(Cy, axis=-1), transpose_a=True) # N 6 1
        with tf.device('/CPU:0'):
            M = tf.matrix_solve(mat, rhs) # N 6 1

        #----- % get R -----
        m1 = tf.squeeze(M[:, 0:3, :], axis=-1) # N 3
        m2 = tf.squeeze(M[:, 3:6, :], axis=-1)

        # --- 
        lambda1 = tf.tile(tf.expand_dims(
            tf.norm(m1, ord='euclidean', axis=1), axis=-1), [1,3])
        lambda2 = tf.tile(tf.expand_dims(
            tf.norm(m2, ord='euclidean', axis=1), axis=-1), [1,3])
        
        self.lambda1 = lambda1

        m1 = m1/lambda1
        m2 = m2/lambda2
        m3 = tf.cross(m1,m2) # N 3
        
        R = tf.stack([m1, m2, m3], axis=1)
        
        with tf.device('/CPU:0'):
            s, U, V = tf.linalg.svd(R)
        R = tf.matmul(U, V, transpose_b=True)
        # print(R[:,0,:].shape)
        M_ = tf.stack([lambda1 * R[:,0,:], lambda2 * R[:,1,:]], axis=1) # N 2 3
    
        return M_

                  
    def q_from_y(self, pts2d, C_IJ_, y_IJ_3d_post, M, y_IJ_post, DM=DM):

        # M: N 2 3
        # y_IJ_3d_post: N 3 m
        # y_IJ_post: N 2 m
        # C_IJ_: 2m 2m

        T = tf.matmul(M, y_IJ_3d_post) # N 2 m

        phi = np.reshape(DM['coeff'], [3, int(DM['numpts']), -1], order='F') # 3 C numcomp

        phi = tf.transpose(phi, perm=[0,2,1]) # 3 numcomp C
        phi_ij = tf.tile(tf.expand_dims(phi, -1), [1,1,1,FLAGS.num_lmks]) - \
                tf.tile(tf.expand_dims(phi, 2), [1,1,FLAGS.num_lmks,1]) # 3 numcomp C C
        phi_IJ = tf.cast(mask_upper_tri_IJ(phi_ij), tf.float32) # 3 numcomp m

        G = tf.transpose(tf.tensordot(M, phi_IJ, axes=[[2], [0]]), perm=[0,3,1,2]) # N m 2 numcomp
        G = tf.reshape(G, [-1, 2*tf.shape(y_IJ_post)[-1], int(DM['numcomp'])]) # N 2m numcomp
        CG = tf.transpose(tf.tensordot(C_IJ_, G, axes=[[1], [1]]), perm=[1,0,2]) # N 2m numcomp
        Cy_T = tf.transpose(tf.tensordot(C_IJ_, 
                tf.reshape(tf.transpose(y_IJ_post - T, perm=[0,2,1]), 
                [-1, 2*tf.shape(y_IJ_post)[-1]]), # N 2m
                axes=[[1], [1]]),
                perm=[1,0])  # N 2m

        # get q
        mat_q = tf.matmul(G, CG, transpose_a=True) # N numcomp numcomp
        rhs_q = tf.matmul(G, tf.expand_dims(Cy_T, axis=-1), transpose_a=True) # N numcomp 1
        with tf.device('/CPU:0'):
            q = tf.matrix_solve(mat_q, rhs_q) # N numcomp 1

        return q


    def zeta_iterative(self, pts2d, cij, DM=DM):
        with tf.variable_scope("pairwise"):

            y3d_0 = np.reshape(DM['mu'], [3, -1], order='F') # 3 C
            
            C_IJ = mask_upper_tri_IJ(cij) # 2 2 m
            C_IJ = tf.linalg.diag(C_IJ) # 2 2 m m
            C_IJ_ = tf.reshape(tf.transpose(C_IJ, perm=[0,2,1,3]), 
                            [2*tf.shape(C_IJ)[-1], 2*tf.shape(C_IJ)[-1]]) # 2m 2m

            y3d = tf.cast(tf.expand_dims(y3d_0, 0), dtype=tf.float32) # 1 3 C
            y3d_N = tf.tile(y3d, [tf.shape(pts2d)[0],1,1]) # N 3 C
            # N 3 C C # N 3 i j
            y_IJ_3d = tf.tile(tf.expand_dims(y3d_N, -1), [1,1,1,FLAGS.num_lmks]) - \
                    tf.tile(tf.expand_dims(y3d_N, 2), [1,1,FLAGS.num_lmks,1])  # N 3 1 C
            # N 3 m
            y_IJ_3d_post = tf.cast(mask_upper_tri_IJ(y_IJ_3d), dtype=tf.float32)

            y = tf.transpose(pts2d, perm=[0,2,1]) # N 2 C            
            # N 2 C C # N 2 i j
            yij = tf.tile(tf.expand_dims(y, -1), [1,1,1,FLAGS.num_lmks]) - \
                    tf.tile(tf.expand_dims(y, 2), [1,1,FLAGS.num_lmks,1])  # N 2 1 C
            # N 2 m
            y_IJ_post = tf.cast(mask_upper_tri_IJ(yij), dtype=tf.float32)

            M = self.M_from_y(pts2d, C_IJ_, y3d_N) # N 2 3
            
            for it in range(5):

                q = self.q_from_y(pts2d, C_IJ_, y_IJ_3d_post, M, y_IJ_post) # N numcomp 1
                q = tf.squeeze(q, axis=-1) # N numcomp

                phi = np.reshape(DM['coeff'], [3, int(DM['numpts']), -1], order='F') # 3 C numcomp
                phi = tf.cast(phi, dtype=tf.float32)
                pts3d_re = tf.tile(y3d, [tf.shape(pts2d)[0],1,1]) + tf.tensordot(
                                q, phi, axes=[[1], [-1]]) # N 3 C

                M = self.M_from_y(pts2d, C_IJ_, pts3d_re) # N 2 3

            pts2d_re = tf.matmul(M, pts3d_re) # N 2 C

            # N 2 C C # N 2 i j
            self.muij = tf.tile(tf.expand_dims(pts2d_re, -1), [1,1,1,FLAGS.num_lmks]) - \
                    tf.tile(tf.expand_dims(pts2d_re, 2), [1,1,FLAGS.num_lmks,1])  # N 2 1 C

            return self.muij, tf.transpose(pts2d_re, perm=[0,2,1]), M

        
    def create_cij(self):
        with tf.variable_scope("pairwise"):
            
            # C(C+1)/2 2 2
            lij = tf.contrib.distributions.fill_triangular(self.lij_) # lower
            lij_diag = tf.nn.relu(tf.matrix_diag_part(lij)) + epsilon # diagonal element >0
            self.lij = tf.matrix_set_diag(lij, lij_diag)
            
            # C(C+1)/2 2 2
            cij = tf.matmul(self.lij, self.lij, transpose_b=True) # lower*upper
            # 2 2 C C
            cij = tf.contrib.distributions.fill_triangular(
                    tf.transpose(cij, perm=[1, 2, 0]))
            cij = tf.matrix_set_diag(cij, tf.zeros([2,2,FLAGS.num_lmks]))
            self.cij = cij + tf.transpose(cij, perm=[0,1,3,2])
            
            # pairwise precision matrix
            sumcij_diag = tf.reduce_sum(self.cij, axis=-1) # 2 2 C
            inv_cov_pair = - self.cij
            self.inv_cov_pair = tf.matrix_set_diag(inv_cov_pair, sumcij_diag)
        return self.cij, self.inv_cov_pair        

        
    def crf_joint(self, mu, inv_cov, labels, muij, cij, inv_cov_pair):
        '''
        mu: N C 2
        cov: N C 2 2
        labels: N C 2
        muij: N 2 C C / N 2 i j
        cij: 2 2 C C
        inv_cov_pair: 2 2 C C
        (mean[3], cov[3], labels, yij, cij, inv_cov_pair)
        '''
        # size
        N = tf.shape(labels)[0]
        C = tf.shape(labels)[1]
        size_mean = [N, 2*C, 1]
        size_cov = [N, 2*C, 2*C]

        outlabels = labels / down_scale

        # N 2 2 C C
        inv_cov_unary = tf.matrix_diag(tf.transpose(inv_cov, perm=[0,2,3,1]))
        precision = inv_cov_unary + tf.tile(tf.expand_dims(inv_cov_pair, 0), 
                                            [N,1,1,1,1])
        # 2 sum cij muij
        # N C2 1
        b = tf.matmul(tf.reshape(tf.transpose(inv_cov_unary, perm=[0,3,1,4,2]),# N C 2 C 2
                                [N,2*C,2*C]), tf.reshape(mu, [N,2*C,1]))
        # N C C 2 2 / N C C 2 1 =>  N C C 2 1
        muc = tf.matmul(tf.tile(tf.expand_dims(tf.transpose(cij, [2,3,0,1]), 0), [N,1,1,1,1]),
                tf.expand_dims(tf.transpose(muij, [0,2,3,1]), -1))
        # N C2 1
        # muc = tf.reshape(tf.transpose(tf.squeeze(tf.reduce_sum(muc, 2)), [0,2,1]), [N,2*C,1])
        muc = tf.reshape(tf.squeeze(tf.reduce_sum(muc, 2)), [N,2*C,1])

        # transpose reshape 
        # N C2 1
        
        b_ = b + muc
        outlabels_ = tf.reshape(outlabels, size_mean)
        
        precision_ = tf.reshape(tf.transpose(precision, perm=[0,3,1,4,2]), size_cov)

        ## cholesky_solve
        ## matrix_inv #LU
        chol = tf.cholesky(precision_)
        mean_ = tf.cholesky_solve(chol, b_)
        # mean_ = tf.linalg.solve(precision_, b_)

        y_diff = tf.subtract(outlabels_, mean_)
        loss = tf.reduce_mean(tf.matmul(tf.matmul(y_diff, precision_, transpose_a=True), y_diff) - \
                2*tf.reduce_sum(tf.log(tf.matrix_diag_part(chol)), -1)) + tf.reduce_mean(
                tf.matmul(y_diff,y_diff, transpose_a=True)) + 1000*tf.reduce_mean(tf.keras.metrics.mae(outlabels_, mean_))
        # tf.add_to_collection('losses', loss)
        return loss, mean_, precision_


    def joint_iterative(self, mean, inv_cov, labels, inv_cov_pair, joint_mean, cij):

        for it in range(5):
            muij_, pts2d_re_, M_ = self.zeta_iterative(pts2d=tf.reshape(joint_mean, 
                                                    tf.shape(labels)), cij=cij)
            loss, joint_mean, precision_ = self.crf_joint(mean, inv_cov, labels, muij_, cij, inv_cov_pair)

        return loss, joint_mean


    def learn_inv_cov(self, var_scope='inv'):
        
        learned_inv_covs = []
        learned_log_det_inv_covs = []

        for i in range(self.num_modules):
            h = tf.concat([self.logits[i], self.last_feature[i]], axis=-1)
            # print(h.shape)
            h = self.conv1_inv(h)
            h_32 = layers.AveragePooling2D(pool_size=(2, 2), strides=2, 
                        name=var_scope+'_AvgPool2D')(h)
            h_32 = self.conv2_inv(h_32)
            h_8 = layers.MaxPool2D(pool_size=(4, 4), strides=4, 
                        name=var_scope+'_MaxPool2D')(h_32) 
            # print(h_8.shape)
            h_8 = tf.transpose(h_8, [0,3,1,2])
            h_8 = tf.reshape(h_8, shape=[tf.shape(h_8)[0], FLAGS.num_lmks, 64])
            h_8 = self.dense1(h_8)
            h_8 = self.dense2(h_8)
            h_8 = self.dense3(h_8)
            
            # N C 2 2 chol
            # with tf.device('/CPU:0'):
            h_low = tf.contrib.distributions.fill_triangular(h_8) # lower
            h_diag = tf.abs(tf.matrix_diag_part(h_low)) + 0.01
            log_h_diag = tf.log(h_diag) # diagonal element >0
            self.h = tf.matrix_set_diag(h_low, h_diag)
            
            # N C 2 2
            learned_inv_cov = tf.matmul(self.h, self.h, transpose_b=True) # lower*upper
            learned_log_det_inv_cov = 2 * tf.reduce_sum(log_h_diag, -1)

            learned_inv_covs.append(learned_inv_cov)
            learned_log_det_inv_covs.append(learned_log_det_inv_cov)
           
        return learned_inv_covs, learned_log_det_inv_covs


    def crf_forward(self, images, labels, is_train=True):
        logits = self.forward(images, is_train=is_train)
        mean = self.compute_mean(logits)
        inv_cov, logdet_invcov= self.learn_inv_cov()
        cij, inv_cov_pair = self.create_cij()
    
        muij_, pts2d_re_, M_ = self.zeta_iterative(pts2d=mean[3], cij=cij)
        
        _, joint_mean, self.precision = self.crf_joint(
            mean[3], inv_cov[3], labels, muij_, cij, inv_cov_pair
            )
        self.loss_joint, self.joint_mean = self.joint_iterative(mean[3], inv_cov[3], 
                                    labels, inv_cov_pair, joint_mean, cij)

        self.loss_unary = self.loss_total_crf_unary(logits, labels, mean, inv_cov, logdet_invcov)
        return self.loss_joint, self.joint_mean, self.precision


    def loss_total_crf_unary(self, output, labels, mean, inv_cov, logdet_invcov):
        """
        unary loss function
        """
        # labels N C 2
        for i in range(len(output)):
            loss = metric.softmax_nll_with_logits(logits=output[i], labels=labels)
            tf.add_to_collection('losses', loss)

            loss_gaussian = 0.5*metric.multi_gaussian_fullcov_nll(mean=mean[i], 
                inv_cov=inv_cov[i], logdet_invcov=logdet_invcov[i], labels=labels)
            tf.add_to_collection('losses', loss_gaussian)
            L1_loss = 2*metric.L1_mean_loss(mean=mean[i], labels=labels)
            tf.add_to_collection('losses', L1_loss)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

