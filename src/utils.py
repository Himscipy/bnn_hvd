import os
import cProfile
import errno
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import time
import matplotlib
from tensorflow import keras
import pickle
import tensorflow_probability as tfp
import sys
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from absl import flags



try:
  import seaborn as sns 
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

tfp_layers = tfp.layers

tf.logging.set_verbosity(tf.logging.INFO)

tfd = tfp.distributions

from matplotlib import figure 
from matplotlib.backends import backend_agg

matplotlib.use("Agg")




class Pre_Post_Process:
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS
    
    def Setup_Seed(self,rank):
        seed_adjustment= rank
        tf.reset_default_graph()
        np.random.seed(6118 + seed_adjustment)
        tf.set_random_seed(1234 + seed_adjustment)
        original_seed = 1092 + seed_adjustment
        seed = tfd.SeedStream(original_seed, salt="random_beta")
        
        return seed


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
    
    def Log_print(self,statement,rank):
        ##
        # The print by rank-0 only
        ##
        assert isinstance(statement,str),"Statement need to be a String datatype..!"
        if rank == 0:
            str_ = "[Rank-{}] ".format(rank)
            print (str_ + statement,flush=True)
        else:
            pass
        return
    
    def plot_weight_posteriors(self,names, qm_vals, qs_vals, fname):

        fig = figure.Figure(figsize=(24, 12))
        canvas = backend_agg.FigureCanvasAgg(fig)
        
        ax = fig.add_subplot(1, 2, 1)

        for n, qm in zip(names, qm_vals):
            sns.distplot(qm.flatten(), ax=ax, label=n)
        ax.set_title("weight means")
        #ax.set_xlim([-1.5, 1.5])
        ax.legend()
        
        ax = fig.add_subplot(1, 2, 2)
        for n, qs in zip(names, qs_vals):
            sns.distplot(qs.flatten(), ax=ax)
        ax.set_title("weight stddevs")
        #ax.set_xlim([0, 1.])
        
        fig.tight_layout()
        canvas.print_figure(fname, format="png")
        #print("saved {}".format(fname))
        return


class Data_API:
    def __init__(self,FLAGS):
        super().__init__()
        self.FLAGS = FLAGS

    
    def load_data(self, rank):
        """Loads CIFAR10 dataset.
        Returns:
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
        #path = '/projects/datascience/hsharma/bnn_horovod/TFP_CIFAR10/RunScript/cifar-10-batches-py'
        #path = '/home/hsharma/WORK/Project_BNN/bnn_horovod/TFP_CIFAR10/cifar-10-batches-py'
        
        if self.FLAGS.DATA_NAME == 'CIFAR-10':
            path = self.FLAGS.DATA_PATH
            print(path)
            num_train_samples = 50000

            x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
            y_train = np.empty((num_train_samples,), dtype='uint8')


            for i in range(1, 6):
                fpath = os.path.join(path, 'data_batch_' + str(i))
                (x_train[(i - 1) * 10000:i * 10000, :, :, :],
                y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

            fpath = os.path.join(path, 'test_batch')
            x_test, y_test = load_batch(fpath)

            y_train = np.reshape(y_train, (len(y_train), 1))
            y_test = np.reshape(y_test, (len(y_test), 1))

            if K.image_data_format() == 'channels_last':
                x_train = x_train.transpose(0, 2, 3, 1)
                x_test = x_test.transpose(0, 2, 3, 1)

            x_test = x_test.astype(x_train.dtype)
            y_test = y_test.astype(y_train.dtype)

            x_train = x_train.astype("float32")
            x_test = x_test.astype("float32")

            x_train /= 255
            x_test /= 255

            if self.FLAGS.subtract_pixel_mean:
                x_train_mean = np.mean(x_train, axis=0)
                x_train -= x_train_mean
                x_test -= x_train_mean
                
            # y_train = y_train.flatten()
            # y_test = y_test.flatten()
            y_train= np.int32(y_train)
            y_test= np.int32(y_test)
        
        else:
            print("ERROR: The dataset is not Available...!")
            return
        return (x_train, y_train), (x_test, y_test)

    def train_input_generator(self, x_train, y_train, batch_size=64):
        assert len(x_train) == len(y_train)
        while True:
            p = np.random.permutation(len(x_train))
            x_train, y_train = x_train[p], y_train[p]
            index = 0
            while index <= len(x_train) - batch_size:
                yield x_train[index:index + batch_size], np.reshape(y_train[index:index + batch_size], -1),
                index += batch_size






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

    def CIFAR10_BNN_model(self):
        """
        CIFAR-10 Basic Model..
        """
        model = tf.keras.Sequential([
            (tfp_layers.Convolution2DFlipout(32,kernel_size=[3,3],activation=tf.nn.relu,padding='SAME',input_shape=self.feature_shape,name='Conv_I')),
            (tfp_layers.Convolution2DFlipout(32,kernel_size=[3,3],activation=tf.nn.relu,padding='SAME',name='Conv_II')),
            (tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', strides=2,name='Max_I_')),
            (tfp_layers.Convolution2DFlipout(64,kernel_size=[3,3],activation=tf.nn.relu,padding='SAME',name='Conv_III')),
            (tfp_layers.Convolution2DFlipout(64,kernel_size=[3,3],activation=tf.nn.relu,padding='SAME',name='Conv_IV')),
            (tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', strides=2,name='Max_II')),
            (tfp_layers.Convolution2DFlipout(128,kernel_size=[3,3],activation=tf.nn.relu,padding='SAME',name='Conv_V')),
            (tfp_layers.Convolution2DFlipout(128,kernel_size=[3,3],activation=tf.nn.relu,padding='SAME',name='Conv_VI')),
            (tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', strides=2,name='Max_III')),
            (tf.keras.layers.Flatten()),
            (tfp_layers.DenseFlipout(128,activation=tf.nn.relu,name='Dense_I')),
            (tfp_layers.DenseFlipout(10,name='Dense_II')),
            ])
        return model