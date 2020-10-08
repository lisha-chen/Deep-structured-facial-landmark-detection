# -*- coding: utf-8 -*-
"""
@author: lisha
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
import tensorflow as tf

from CNNmodels import FAN_crf_pred


import scipy.io

#%%----------------------------------------------------------------


FLAGS = tf.app.flags.FLAGS

# configurations

tf.app.flags.DEFINE_integer("batch_size",
                    default=1,
                    help="Batch size.")
tf.app.flags.DEFINE_integer("gt_num_lmks",
                    default=68,
                    help="Number of landmarks in ground truth")
tf.app.flags.DEFINE_integer("num_lmks",
                    default=68,
                    help="Number of landmarks in prediction")
tf.app.flags.DEFINE_integer("eval_num",
                    default=689,
                    help="Number of evaluation faces.")
tf.app.flags.DEFINE_float("offset",
                    default=0.,
                    help="Offset to add to prediction.")
# directories

tf.app.flags.DEFINE_string('data_eval_dir',
    './data/300w_train_val/val/',
    help=""" eval data folder""")
tf.app.flags.DEFINE_string('eval_tfrecords_file',
    'thrWtrain_val_689.tfrecords',
    help=""" eval tfrecords file""")

tf.app.flags.DEFINE_string('model_dir',
    './pretrained_models/300wtrain/',
    """Directory for model file""")
tf.app.flags.DEFINE_string('model_name',
    'model_300wtrain.ckpt',
    """model file name""")
tf.app.flags.DEFINE_string('facemodel_path',
    './facemodel/DM68_wild34.mat',
    # './facemodel/DM68_lp34.mat',
    """face model path""")
tf.app.flags.DEFINE_string('save_result_name',
    './results/300wtrain_val689.mat',
    """mat file path to save result""")
#%%----------------------------------------------------------------
def main(argv=None):

    model_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)

    FAN_crf_model = FAN_crf_pred.FAN_crf_eval(model_path = model_path, FLAGS=FLAGS)
    preds, precision, labels = FAN_crf_model.predict_tfrecords()
    scipy.io.savemat(FLAGS.save_result_name, 
                  {"joint_mean":preds,
                  "inv_cov":precision,
                  "labels":labels})

if __name__ == '__main__':
    tf.app.run()









