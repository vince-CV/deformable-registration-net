import numpy as np
import tensorflow as tf
from WarpST import WarpST
from resnet_model import imagenet_resnet_v2
from ops import *

class CNN(object):
  
  def __init__(self, name, is_train):
    self.name = name
    self.is_train = is_train
    self.reuse = None
  
  def __call__(self, x):
    with tf.variable_scope(self.name, reuse=self.reuse):
      
      print("x shape before in CNN: ", x.get_shape())
      x = conv2d(x, "conv1", 64, 3, 1, "SAME", True, tf.nn.elu, self.is_train)   # 3*3 convolutions layer (with 0 padding)
      
      x = tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1], "SAME")                        # 2*2 downsampling layer

      x = conv2d(x, "conv2", 128, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
      x = conv2d(x, "out1", 128, 3, 1, "SAME", True, tf.nn.elu, self.is_train)

      x = tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1], "SAME")
      
      x = conv2d(x, "out2", 2, 3, 1, "SAME", False, None, self.is_train)

    if self.reuse is None:
      self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      self.saver = tf.train.Saver(self.var_list)
      self.reuse = True

    return x

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)

class DIRNet(object):
  def __init__(self, sess, config, name, is_train):
    self.sess = sess
    self.name = name
    self.is_train = is_train  

    im_shape = [config.batch_size] + config.im_size + [3]  # [1] for single channel

    self.x = tf.placeholder(tf.float32, im_shape)
    self.y = tf.placeholder(tf.float32, im_shape)
    
    # concating moving and fixed images
    self.xy = tf.concat([self.x, self.y], 3)

    self.vCNN = CNN("vector_CNN", is_train=self.is_train)

    # CNN regressor
    # the output is the local deformation parameters for the spatial transformer
    self.v = self.vCNN(self.xy)   
    
    # spatial transformer generates a dense DVF;
    # resampler to warp the moving image to the fixed image using DVF;
    self.z = WarpST(self.x, self.v, config.im_size) 
    
    if self.is_train:
      self.loss = ncc(self.y, self.z)
      #self.loss = mse(self.y, self.z)
      
      global_step = tf.Variable(0, trainable=False)
    
      learning_rate = tf.train.exponential_decay(config.lr, global_step=global_step, decay_steps=10, decay_rate=0.99)
      
      self.optim = tf.train.AdamOptimizer(learning_rate)

      self.train = self.optim.minimize( -self.loss, global_step=global_step, var_list=self.vCNN.var_list)

    #self.sess.run(tf.variables_initializer(self.vCNN.var_list))
    self.sess.run(tf.global_variables_initializer())

  def fit(self, batch_x, batch_y):
    _, loss = \
      self.sess.run([self.train, self.loss], {self.x:batch_x, self.y:batch_y})
    return loss

  def deploy(self, dir_path, x, y):
    z = self.sess.run(self.z, {self.x:x, self.y:y})
    for i in range(z.shape[0]):
      save_image_with_scale(dir_path+"/{:02d}_x.JPG".format(i+1), x[i,:,:,0])
      save_image_with_scale(dir_path+"/{:02d}_y.JPG".format(i+1), y[i,:,:,0])
      save_image_with_scale(dir_path+"/{:02d}_z.JPG".format(i+1), z[i,:,:,0])

  def save(self, dir_path):
    self.vCNN.save(self.sess, dir_path+"/model.ckpt")

  def restore(self, dir_path):
    self.vCNN.restore(self.sess, dir_path+"/model.ckpt")

class ResNet(object):

    def __init__(self, sess, config, name, is_train):
        self.sess = sess
        self.name = name
        self.is_train = is_train

        # image shape for grayscale images
        im_shape = [config.batch_size] + config.im_size + [1]  

        # x => moving image, y => fixed image
        self.x = tf.placeholder(tf.float32, im_shape)   
        self.y = tf.placeholder(tf.float32, im_shape)

        self.labels = tf.placeholder(tf.int32, [config.batch_size])

        # x and y concatenated in color channel
        self.xy = tf.concat([self.x, self.y], 3)

        self.model = imagenet_resnet_v2(18, 2, data_format='channels_first')
        self.v = self.model(self.xy, is_training=True)
        print("Size of V: ", self.v)
        self.z = WarpST(self.x, self.v, config.im_size)
       

        # create predictions => filter highest likelyhood from logits
        #self.prediction = tf.argmax(self.logits, 1)
        #self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #self.saver = tf.train.Saver(self.var_list)

        if self.is_train:
            #loss definition and weighting of the 2 loss functions
            #self.loss = - self.disease_loss(self.labels, self.logits)
            self.loss = mse(self.y, self.z)

            self.optim = tf.train.AdamOptimizer(config.lr)
            self.train = self.optim.minimize( - self.loss)

        # self.sess.run(tf.variables_initializer(self.vCNN.var_list))
        self.sess.run(tf.global_variables_initializer())

    def save(self, dir_path):
        self.saver.save(self.sess, dir_path + "/model_class.ckpt")

    def restore(self, dir_path):
        self.saver.restore(self.sess, dir_path + "/model_class.ckpt")

    def fit(self, batch_x, batch_y, batch_labels):
        _, loss, pred = \
            self.sess.run([self.train, self.loss, self.prediction], {self.x: batch_x, self.y: batch_y, self.labels: batch_labels})
        return loss, pred

    def disease_loss(self, labels, logits):
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        loss = tf.losses.softmax_cross_entropy( onehot_labels=onehot_labels, logits=logits)
        return loss

    def deploy_with_labels(self, x, y, labels):
        
        pred = self.sess.run([self.prediction], {self.x: x, self.y: y})

        pred = int(pred[0])

        return pred

