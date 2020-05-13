import os
import cProfile
import errno
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import time
from tensorflow import keras
import pickle
import tensorflow_probability as tfp
import sys
from absl import flags

tfp_layers = tfp.layers

tf.logging.set_verbosity(tf.logging.INFO)

tfd = tfp.distributions




class Pre_Post_Process:
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS
    
    def create_config_proto(self):
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = self.FLAGS.num_intra
        config.inter_op_parallelism_threads = self.FLAGS.num_inter
        config.allow_soft_placement         = True
        os.environ['KMP_BLOCKTIME'] = str(self.FLAGS.kmp_blocktime)
        os.environ['KMP_AFFINITY'] = self.FLAGS.kmp_affinity
        return config

    def train_input_generator(self,x_train, y_train, batch_size=64):
        assert len(x_train) == len(y_train)
        while True:
            p = np.random.permutation(len(x_train))
            x_train, y_train = x_train[p], y_train[p]
            index = 0
            while index <= len(x_train) - batch_size:
                yield x_train[index:index + batch_size], \
                    y_train[index:index + batch_size],
                index += batch_size

    def Write_TrainingLog(self,dirmake):
        with open( os.path.join( dirmake, "training.log"), 'w') as _out:
                total_parameters = 0
                for variable in tf.trainable_variables():
                    this_variable_parameters = np.prod([s for s in variable.shape])
                    total_parameters += this_variable_parameters
                    _out.write("{} has shape {} and {} total paramters to train.\n".format(
                        variable.name,
                        variable.shape,
                        this_variable_parameters
                    ))
                    _out.write( "Total trainable parameters for this network: {} \n".format(total_parameters))


class Model_CNN_BNN:
    def __init__(self,feature_shape,Num_class):
        super().__init__()
        self.feature_shape = feature_shape
        self.Num_class = Num_class

    def BNN_conv_model(self):
        """1-layer convolutionflipout model  and 
        last layer as dense
        """
        
    #   Define the Model structure 
        model = tf.keras.Sequential([
            (tf.keras.layers.Reshape(self.feature_shape)),
            (tfp_layers.Convolution2DFlipout(64,kernel_size=[5,5],activation=tf.nn.relu,padding='SAME',input_shape=self.feature_shape)), 
            (tf.keras.layers.MaxPooling2D(pool_size=[2, 2],strides=[2, 2],padding='SAME')),
            (tf.keras.layers.Flatten()),
            (tf.keras.layers.Dense(self.Num_class))])
        return model


    def BNN_FC_model(self,):
        """1-layer Denseflipout model and 
        last layer as simpledense 
        """
    #   Define the Model structure 
        model = tf.keras.Sequential([
            (tfp_layers.DenseFlipout(64,activation=tf.nn.relu,input_shape=(28*28,))),
            (tf.keras.layers.Dense(self.Num_class))])
        return model

    def BNN_conv_model_nonFlip(self,):
        """1-layer convolutionflipout model  and 
        last layer as dense
        """
        
    #   Define the Model structure 
        model = tf.keras.Sequential([
            (tf.keras.layers.Reshape(self.feature_shape)),
            (tfp_layers.Convolution2DReparameterization(64,kernel_size=[5,5],activation=tf.nn.relu,padding='SAME')), 
            (tf.keras.layers.MaxPooling2D(pool_size=[2, 2],strides=[2, 2],padding='SAME')),
            (tf.keras.layers.Flatten()),
            (tf.keras.layers.Dense(self.Num_class))])
        return model


    def BNN_FC_model_nonFlip(self):
        """1-layer Denseflipout model and 
        last layer as simpledense 
        """
    #   Define the Model structure 
        model = tf.keras.Sequential([
            (tfp_layers.DenseReparameterization(64,activation=tf.nn.relu,input_shape=(28*28,))),
            (tf.keras.layers.Dense(self.Num_class))])
        return model



    def CNN_conv_model(self):
        """1-layer convolution model with  and 
        last layer as simple dense
        """
    #   Define the Model structure 
        model = tf.keras.Sequential([
            (tf.keras.layers.Reshape(self.feature_shape)),
            (tf.keras.layers.Conv2D(filters=64,kernel_size=[5, 5],
                                    activation=tf.nn.relu,
                                    padding="SAME",
                                    )),   
            (tf.keras.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2],padding="SAME")),    
            (tf.keras.layers.Flatten()),
            (tf.keras.layers.Dense(self.Num_class))])
        return model

    def CNN_FC_model(self):
        """2-layer convolution model with second last and 
        last layer as Bayesian 
        """
    #   Define the Model structure 
        model = tf.keras.Sequential([
            (tf.keras.layers.Dense(64,activation=tf.nn.relu,input_shape=(28*28,))),
            (tf.keras.layers.Dense(self.Num_class))])
        return model



