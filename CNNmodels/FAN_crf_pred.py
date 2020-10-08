
import os


import numpy as np
import tensorflow as tf


from CNNmodels import FAN_crf as FAN
import scipy
import scipy.io


#%%----------------------------------------------------------------


FLAGS = tf.app.flags.FLAGS

# hyperparameters

tf.app.flags.DEFINE_list("IMAGE_SIZE",
					default=[256, 256],
					help="Input image size.")

tf.app.flags.DEFINE_float("gpu_mem_fraction",
					default=0.25,
					help="GPU memory fraction to use.")


#%%----------------------------------------------------------------





class FAN_crf_pred():
	def __init__(self, model_path, FLAGS=FLAGS):
		self.graph = tf.Graph() 
		self.args = FLAGS
		with self.graph.as_default():
			self.images = tf.compat.v1.placeholder(tf.float32, 
				shape=(None,self.args.IMAGE_SIZE[0],self.args.IMAGE_SIZE[1],3))
			self.CNN_FA = FAN.FAN(FLAGS=self.args, num_modules=int(4))
			self.joint_mean, self.precision = self.CNN_FA.crf_forward_predict(self.images, is_train=False)
			saver = tf.train.Saver(var_list=tf.global_variables())

			self.gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self.args.gpu_mem_fraction)   
			self.sess = tf.Session(config=tf.compat.v1.ConfigProto(gpu_options=self.gpu_options)) 
			with self.sess.as_default():
				saver.restore(tf.get_default_session(), model_path)
				
	def predict_single(self, img_string):
		# Run forward pass to calculate joint mean and precision 
		image_string = tf.read_file(img_string)
		face_ = tf.image.decode_image(tf.reshape(image_string, shape=[]), channels=3 ,expand_animations = False)
		face_ = tf.image.convert_image_dtype(face_, dtype=tf.float32)
		h = tf.shape(face_)[0]
		w = tf.shape(face_)[1]
		face_ = tf.image.resize(tf.expand_dims(face_, 0), [self.args.IMAGE_SIZE[0],self.args.IMAGE_SIZE[1]])
		with tf.Session(config=tf.compat.v1.ConfigProto(gpu_options=self.gpu_options)).as_default():
			face = face_.eval()
			ori_h = h.eval()
			ori_w = w.eval()
		feed_dict = {self.images: face} 
		joint_mean, precision = self.sess.run([self.joint_mean, self.precision], feed_dict=feed_dict)
		joint_mean = np.reshape(joint_mean, [68, 2])
		joint_mean[:, 0] = joint_mean[:, 0] * (ori_h-1)/(self.args.IMAGE_SIZE[0]-1) *255./63. +1
		joint_mean[:, 1] = joint_mean[:, 1] * (ori_w-1)/(self.args.IMAGE_SIZE[1]-1) *255./63. +1
		return joint_mean, precision


	def predict_single_bbox(self, img_string, bbox):
		# bbox left, top, right, bottom
		face, ori_h, ori_w, scale_y, scale_x  = self.crop_resize_img(img_string, bbox)
		feed_dict = {self.images: face} 
		joint_mean, precision = self.sess.run([self.joint_mean, self.precision], feed_dict=feed_dict)
		joint_mean = np.reshape(joint_mean, [68, 2])
		joint_mean[:, 0] = joint_mean[:, 0]/scale_x *255./63.+bbox[0] +1
		joint_mean[:, 1] = joint_mean[:, 1]/scale_y *255./63.+bbox[1] +1
		return joint_mean, precision


	def crop_resize_img(self, img_string, bbox):

		image_string = tf.read_file(img_string)
		img_ = tf.image.decode_image(tf.reshape(image_string, shape=[]), channels=3 ,expand_animations = False)
		img_ = tf.image.convert_image_dtype(img_, dtype=tf.float32)
		h = tf.shape(img_)[0]
		w = tf.shape(img_)[1]

		xmin = bbox[0]
		ymin = bbox[1]
		xmax = bbox[2]
		ymax = bbox[3]

		bbox_h = ymax - ymin
		bbox_w = xmax - xmin

		y1 = ymin / tf.cast(h - 1, tf.float32)
		x1 = xmin / tf.cast(w - 1, tf.float32)
		y2 = ymax / tf.cast(h - 1, tf.float32)
		x2 = xmax / tf.cast(w - 1, tf.float32)
		
		img_ = tf.squeeze(tf.image.crop_and_resize(tf.expand_dims(img_, 0),
						boxes=[[y1, x1, y2, x2]], box_ind=[0],
						crop_size=[self.args.IMAGE_SIZE[0], self.args.IMAGE_SIZE[1]], 
						method='bilinear'), axis=0)
		
		scale_y_ = (self.args.IMAGE_SIZE[1]-1) / ((x2 - x1) * tf.cast(w - 1, tf.float32))
		scale_x_ = (self.args.IMAGE_SIZE[0]-1) / ((y2 - y1) * tf.cast(h - 1, tf.float32))

		with tf.Session(config=tf.compat.v1.ConfigProto(gpu_options=self.gpu_options)).as_default():
			
			img = img_.eval()
			ori_h = h.eval()
			ori_w = w.eval()

			scale_y = scale_y_.eval()
			scale_x = scale_x_.eval()
		if img.ndim < 4:
			img = np.expand_dims(img, axis=0)
		return img, ori_h, ori_w, scale_y, scale_x





class FAN_crf_eval():
	def __init__(self, model_path, FLAGS=FLAGS):
		self.graph = tf.Graph() 
		self.args = FLAGS
		with self.graph.as_default():
			self.eval_dataset = tf.data.TFRecordDataset(os.path.join(self.args.data_eval_dir,
														self.args.eval_tfrecords_file))
			self.eval_dataset = self.eval_dataset.map(map_func=self.read_file).batch(self.args.batch_size)
			self.eval_iterator = self.eval_dataset.make_one_shot_iterator()
			
			self.images, self.labels, self.corner, self.scale = self.eval_iterator.get_next()

			self.CNN_FA = FAN.FAN(FLAGS=self.args, num_modules=int(4))

			self.joint_mean, self.precision = self.CNN_FA.crf_forward_predict(self.images, is_train=False)
			self.pred_lmk = tf.reshape(self.joint_mean, [-1, self.args.num_lmks, 2])
			self.pred_lmk = self.pred_lmk*255./63./self.scale + self.corner + 1.5 + self.args.offset

			saver = tf.train.Saver(var_list=tf.global_variables())

			self.gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self.args.gpu_mem_fraction)   
			self.sess = tf.Session(config=tf.compat.v1.ConfigProto(gpu_options=self.gpu_options)) 
			with self.sess.as_default():
				saver.restore(tf.get_default_session(), model_path)


	def predict_tfrecords(self):

		joint_mean_list = []
		labels_list = []
		inv_cov_list = []
		for i in range(self.args.eval_num):
			pred_lmk, precision, labels_ = self.sess.run([self.pred_lmk, self.precision, self.labels])
			print(i)               
			
			joint_mean_list.append(pred_lmk)
			labels_list.append(labels_)
			inv_cov_list.append(precision)
		
		joint_means = np.squeeze(np.array(joint_mean_list))
		inv_covs = np.squeeze(np.array(inv_cov_list))
		labels = np.squeeze(np.array(labels_list))
		return joint_means, inv_covs, labels
		

	def read_file(self, example):

		example_fmt = {
			"imagename": tf.FixedLenFeature((), tf.string, ""),
			"pts2d_68": tf.FixedLenFeature((self.args.gt_num_lmks*2), tf.float32, 
						default_value=list(np.zeros(self.args.gt_num_lmks*2)))
			}

		parsed = tf.parse_single_example(example, example_fmt)
		
		label = tf.reshape(parsed["pts2d_68"], [-1, 2])
		
		img_string = tf.read_file(tf.reduce_join([tf.constant(
					self.args.data_eval_dir), parsed["imagename"]]))
		
		image = tf.image.decode_image(tf.reshape(img_string, shape=[]), channels=3)
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		h = tf.shape(image)[0]
		w = tf.shape(image)[1]
		
		label = label-1
		h = tf.shape(image)[0]
		w = tf.shape(image)[1]
		
		xmin = tf.reduce_min(label[:,0], axis=0)
		ymin = tf.reduce_min(label[:,1], axis=0)
		xmax = tf.reduce_max(label[:,0], axis=0)
		ymax = tf.reduce_max(label[:,1], axis=0)

		bbox_h = ymax - ymin
		bbox_w = xmax - xmin

		scale_factor = 0.01
		y1 = (ymin-bbox_h*scale_factor) / tf.cast(h - 1, tf.float32)
		x1 = (xmin-bbox_w*scale_factor) / tf.cast(w - 1, tf.float32)
		y2 = (ymax+bbox_h*scale_factor) / tf.cast(h - 1, tf.float32)
		x2 = (xmax+bbox_w*scale_factor) / tf.cast(w - 1, tf.float32)
		
		image = tf.squeeze(tf.image.crop_and_resize(tf.expand_dims(image, 0),
						boxes=[[y1, x1, y2, x2]], box_ind=[0],
						crop_size=[self.args.IMAGE_SIZE[0], self.args.IMAGE_SIZE[1]], 
						method='bilinear'), axis=0)
		
		corner = tf.stack([x1 * tf.cast(w - 1, tf.float32) * tf.ones(self.args.num_lmks), 
						   y1 * tf.cast(h - 1, tf.float32) * tf.ones(self.args.num_lmks)], axis=1)
		
		scale = tf.stack([tf.ones(self.args.num_lmks, dtype=tf.float32) * (self.args.IMAGE_SIZE[1]-1) \
						  / ((x2 - x1) * tf.cast(w - 1, tf.float32)), 
						   tf.ones(self.args.num_lmks, dtype=tf.float32) * (self.args.IMAGE_SIZE[0]-1) \
						  / ((y2 - y1) * tf.cast(h - 1, tf.float32))], axis=1)

		image.set_shape([self.args.IMAGE_SIZE[0], self.args.IMAGE_SIZE[1], 3])

		return image, label+1, corner, scale



