import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os
import h5py
from random import Random
import time
import tensorflow.keras.backend as K
import pickle
from tensorflow.python.training import session_run_hook
import tensorflow_probability as tfp

try:
  import seaborn as sns 
  sns.set_context("talk")
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False
from matplotlib import figure 
from matplotlib.backends import backend_agg


tfd= tfp.distributions
layers = tf.keras




scale_factor = 1.
spin_scale_factor = 1.


def Setup_Seed(params):

    if params.hvd:
        import horovod.tensorflow as hvd
        hvd.init()
        #EJ NB to set seeds here
        seed_adjustment= hvd.rank()
        tf.reset_default_graph()
        np.random.seed(6118 + seed_adjustment)
        tf.set_random_seed(1234 + seed_adjustment)
        original_seed = 1092 + seed_adjustment
        seed = tfd.SeedStream(original_seed, salt="random_beta")
        #print('Seed Initialized')
        #print("Original_seed:",original_seed,"Seed",seed())
    else:
        seed_adjustment= 0
        tf.reset_default_graph()
        np.random.seed(6118 + seed_adjustment)
        tf.set_random_seed(1234 + seed_adjustment)

        original_seed = 1092 + seed_adjustment
        seed = tfd.SeedStream(original_seed, salt="random_beta")
    
    return seed
        




def create_config_proto(params):
   '''EJ: TF config setup'''
   config = tf.ConfigProto()
   config.intra_op_parallelism_threads = params.num_intra
   config.inter_op_parallelism_threads = params.num_inter
   config.allow_soft_placement         = True
   os.environ['KMP_BLOCKTIME'] = str(params.kmp_blocktime)
   os.environ['KMP_AFFINITY'] = params.kmp_affinity
   if params.hvd:
       import horovod.tensorflow as hvd
       hvd.init()
       if hvd.rank()==0: print("-----------Imported hvd-----------")
       config.gpu_options.visible_device_list = str(hvd.local_rank())
   return config


class Dataset(object):
    def __init__(self, args, SNR=1.0, shift=True, SEED=None,
                 real_noise=False, zero=False,
                 model_name='spin_omegas_model1_shift_data_trained',
                 data_path='/projects/mmaADSP/elise/hongyu/BNN/singlescript_gw/',
                 save_path='/projects/datascience/hsharma/GW_Project/Single_Script/data_store',
                 Test_data = '/projects/mmaADSP/hsharma/',
                 if_run_test=False):
        self.seed_set = SEED
        self.SNR = SNR
        self.shift = shift
        self.real_noise = real_noise
        self.zero = zero
        self.model_name = model_name
        self.data_path = data_path
        self.save_path = save_path
        self.test_dataPath = Test_data
        self.load_h5()
        self.Load_Do_Shuffling()

        self.TFseed=args.TFseed
        if args.hvd:
            import horovod.tensorflow as hvd
            hvd.init()
            self.hvd=True
            self.num_workers=hvd.size()
            self.rank=hvd.rank()
            if hvd.rank() == 0:
                self.make_folder()
        else:
            self.hvd=False
            self.make_folder()

       
    def make_folder(self):
        
        self.final_save_path = os.path.join(self.save_path, self.model_name)

        if os.path.exists(self.final_save_path):
            pass
            #print("Directory already exists: %s " % self.final_save_path)
        else:
            try:
                os.mkdir(self.final_save_path)
            except OSError:
                print("Creation of the directory: %s failed" % self.final_save_path)
            else:
                
                print("Successfully created the directory: %s " % self.final_save_path) 
        
    
    def load_Shuffled_test_datah5(self,filepath=None):
        if filepath is None:
            filepath = self.test_dataPath +'SNR_interval_TestData_.h5'
        
        self.h5_file = h5py.File(filepath, 'r')
        self.h5_keys = self.h5_file.keys()
        self.test_data_ready = self.h5_file[u'test_data']
        self.test_label_ready = self.h5_file[u'test_label']            


    def load_h5(self, filepath=None):
        #
        # using smaller test data to train since orignal
        # Training data no longer exist on Theta in the mmadsp project.
        #
        if filepath is None:
            filepath = self.test_dataPath +'SNR_interval_TestData_.h5'
            
        self.h5_file = h5py.File(filepath, 'r')
        self.h5_keys = self.h5_file.keys()
        self.train_data_ready = self.h5_file[u'test_data']
        self.train_label_ready = self.h5_file[u'test_label']
     


    def Load_Do_Shuffling(self):
        (self.train_data_shuffled, self.train_label_shuffled) = self.shuffling(0,self.train_data_ready, self.train_label_ready)


    def shuffling(self,i, A, B):
        C = list(zip(A, B))
        #EJ: don't use global function shuffle
        #print("Check rng",i,Random().random())
        Random(self.seed_set()).shuffle(C) # seed Added to the Random for reproducability.

        A, B = zip(*C)
        return (np.asarray(A), np.asarray(B))

    def get_Test_batch(self, index, batch_size):

        (test_data_batch, test_output_batch) = (self.test_data_ready[(index+8) * batch_size : (index+9) * batch_size, :], self.test_label_ready[(index+8) * batch_size : (index+9) * batch_size, :])
        
        test_output_batch = test_output_batch / scale_factor
        test_output_batch[:, 2] = test_output_batch[:, 2] / spin_scale_factor
        
        return (test_data_batch, test_output_batch)

    
    def get_batch_version2(self,index,batch_size):
        num_batches = int(self.train_data_ready.shape[0] / batch_size)
        real_index = index % num_batches

        train_data_batch = self.train_data_shuffled[int(real_index * batch_size):int((real_index + 1) * batch_size), :]
        train_label_batch =self.train_label_shuffled[int(real_index * batch_size):int((real_index + 1) * batch_size), :] 

        return (train_data_batch, train_label_batch)
 


    def get_batch(self,index,batch_size):
        #EJ: new batch generation, data set already in memory
        num_batches = int(self.train_data_ready.shape[0] / batch_size)
        real_index = index % num_batches

        if real_index == 0:
            (self.train_data_shuffled, self.train_label_shuffled) = self.shuffling(real_index,self.train_data_ready, self.train_label_ready)

        train_data_batch = self.train_data_shuffled[int(real_index * batch_size):int((real_index + 1) * batch_size), :]
        train_label_batch =self.train_label_shuffled[int(real_index * batch_size):int((real_index + 1) * batch_size), :] 

        return (train_data_batch, train_label_batch)
        
    def train_batch(self, global_index, index, batch_size):
        #EJ use global index in the case of re-starts. 
        if (global_index+1) > 9000 and (global_index+1) <= 18000:
            if hvd.rank() == 0 and index == 0 :
                print("Reading data: SNR_interval_1.h5")

            self.load_h5(filepath=self.data_path+"SNR_interval_1.h5")
        if (global_index+1) > 18000 and (global_index+1) <= 27000:
            if hvd.rank() == 0 and index == 0:
                print("Reading data: SNR_interval_2.h5")
            
            self.load_h5(filepath=self.data_path+"SNR_interval_2.h5")
        if (global_index+1) > 27000 and (global_index+1) <= 36000:
            if hvd.rank() == 0 and index == 0:
                print("Reading data: SNR_interval_3.h5")

            self.load_h5(filepath=self.data_path+"SNR_interval_3.h5")
        if (global_index+1) > 36000 and (global_index+1) <= 48000:
            if hvd.rank() == 0 and index == 0:
                print("Reading data: SNR_interval_4.h5")
            
            self.load_h5(filepath=self.data_path+"SNR_interval_4.h5")
        if (global_index+1) > 48000 and (global_index+1) <= 60000:
            if hvd.rank() == 0 and index == 0:
                print("Reading data: SNR_interval_5.h5")
            
            self.load_h5(filepath=self.data_path+"SNR_interval_5.h5")
        if (global_index+1) > 60000:
            if hvd.rank() == 0 and index == 0:
                print("Reading data: SNR_interval_6.h5")
            
            self.load_h5(filepath=self.data_path+"SNR_interval_6.h5")
            
        
        train_x, train_y = self.get_batch(index, batch_size)
        return (train_x, train_y)
    
    def train_batch_Flag (self, flag, index, batch_size):
        if flag == 0:
            # if self.hvd:
            #     print('HVD_RANK {} "Running SNR_interval_0'.format(hvd.rank()))
            # else:
            #     print('RANK 0 "Running SNR_interval_0')    
            pass
        
              
        train_x , train_y = self.get_batch_version2(index,batch_size)

        # train_x, train_y = self.get_batch(index, batch_size)
        return (train_x, train_y)

    def Fake_train_batch(self,batch_size):
        signal_len = 8192
        x_train = np.random.rand(batch_size,signal_len).astype(np.float32)
        y_train = np.random.rand(batch_size,5).astype(np.int32)
        return (x_train, y_train)
    
    def test_batch(self,batch_size):
        print("Reading Shuffled Test data:")
        self.load_Shuffled_test_datah5(filepath=self.test_dataPath+"SNR_interval_TestData_.h5")

        # Index is hard coded here to 0, in the original code the index can range from 0-200
        # to generate different sets. of training batches. (original code: function_spin_1Update.py , see  test_batch function impelementation)
        test_x, test_y = self.get_Test_batch(0, batch_size)
        return (test_x, test_y)


class BNN_Train_Model(tf.keras.Model):

    def __init__(self,SEED=None):
        self.seed_set = SEED

        super(Train_Model, self).__init__()
        self.signal_length = 8192
        self.reshape1 = tf.keras.layers.Reshape((self.signal_length, 1, 1,), input_shape=(self.signal_length,))

        self.conv1 = tfp.layers.Convolution2DFlipout(64, [16, 1], padding='VALID', activation=tf.nn.relu, seed=self.seed_set(),dilation_rate=(1,1))

        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=[4, 1], strides=[4, 1])
        
        self.conv2 = tfp.layers.Convolution2DFlipout(128, [16, 1], padding='VALID', activation=tf.nn.relu, seed=self.seed_set(), dilation_rate=(1,1))
        
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=[4, 1], strides=[4, 1])            
        
        self.conv3 = tfp.layers.Convolution2DFlipout(256, [16, 1], padding='VALID', activation=tf.nn.relu, seed=self.seed_set(), dilation_rate=(2,1))

        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=[4, 1], strides=[4, 1])

        self.conv4 = tfp.layers.Convolution2DFlipout(256, [32, 1], padding='SAME', activation=tf.nn.relu, seed=self.seed_set(),dilation_rate=(2,1))
        
        self.maxpool4 = tf.keras.layers.MaxPool2D(pool_size=[4, 1], strides=[4, 1])
        #print(maxpool4)

        self.conv3_1_1 = tfp.layers.Convolution2DFlipout(128, [4, 1], padding='SAME', activation=tf.nn.relu,seed=self.seed_set(),dilation_rate=(2,1))
        self.conv4_1_1 = tfp.layers.Convolution2DFlipout(128, [4, 1], padding='SAME', activation=tf.nn.relu,seed=self.seed_set(),dilation_rate=(2,1))
        self.conv5_1_1 = tfp.layers.Convolution2DFlipout(64, [2, 1], padding='SAME', activation=tf.nn.relu,seed=self.seed_set(),dilation_rate=(2,1))
        
      
        self.flatten = tf.keras.layers.Flatten()

        #
        # Bayesian layers
        # #
        self.linear1_1 = tfp.layers.DenseFlipout(128,  activation=tf.nn.relu,seed=self.seed_set(), name='linear1_1')
        self.linear1_2 = tfp.layers.DenseFlipout(128,  activation=tf.nn.relu,seed=self.seed_set(), name='linear1_2')
        self.linear1_3 = tfp.layers.DenseFlipout(128,  activation=tf.nn.relu,seed=self.seed_set(), name='linear1_3')

        self.linear2_1 = tfp.layers.DenseFlipout(128, activation=None,seed=self.seed_set(), name='linear2_1')
        self.linear2_2 = tfp.layers.DenseFlipout(128, activation=None,seed=self.seed_set(), name='linear2_2')
        self.linear2_3 = tfp.layers.DenseFlipout(128, activation=None,seed=self.seed_set(), name='linear2_3')

            
        self.final_1 = tfp.layers.DenseFlipout( 1,  activation=tf.nn.tanh,seed=self.seed_set(), name='final_1') # or softplus
        self.final_2 = tfp.layers.DenseFlipout( 1,  activation=tf.nn.tanh,seed=self.seed_set(), name='final_2') # or softplus
        self.final_3 = tfp.layers.DenseFlipout( 1,  activation=tf.nn.tanh,seed=self.seed_set(), name='final_3')

        
        self.concat = tf.keras.layers.Concatenate(axis=1)

    
    
    def call(self, inputs):
        x = self.reshape1(inputs)
        x = self.conv1(x)
        x= self.maxpool1(x) 
        x= self.conv2(x) 
        x= self.maxpool2(x)
        x=self.conv3(x)
        x=self.maxpool3(x)
        x=self.conv4(x)
        x=self.maxpool4(x)
        x =self.conv3_1_1(x)
        x =self.conv4_1_1(x)
        x =self.conv5_1_1(x)
        
        x1=x
        x2=x
        x3=x

        x1 =self.flatten(x1)
        x2 =self.flatten(x2)
        x3 =self.flatten(x3)
        
        x1=self.linear1_1(x1)
        x2=self.linear1_2(x2)
        x3=self.linear1_3(x3)
        
        x1 = self.linear2_1(x1)
        x2 = self.linear2_2(x2)
        x3 = self.linear2_3(x3)

        x1=self.final_1(x1)
        x2=self.final_2(x2)
        x3=self.final_3(x3)

        outputs = self.concat([x1, x2, x3]) 

        return outputs
   

class Train_Model(tf.keras.Model):

    def __init__(self,SEED=None):
        self.seed_set = SEED

        super(Train_Model, self).__init__()
        self.signal_length = 8192
        self.reshape1 = tf.keras.layers.Reshape((self.signal_length, 1, 1,), input_shape=(self.signal_length,))

        self.conv1 = tf.keras.layers.Conv2D(64, [16, 1], padding='VALID', activation=tf.nn.relu, dilation_rate=(1,1))

        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=[4, 1], strides=[4, 1])
        
        self.conv2 = tf.keras.layers.Conv2D(128, [16, 1], padding='VALID', activation=tf.nn.relu, dilation_rate=(2,1))
        
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=[4, 1], strides=[4, 1])            
        
        self.conv3 = tf.keras.layers.Conv2D(256, [16, 1], padding='VALID', activation=tf.nn.relu, dilation_rate=(2,1))

        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=[4, 1], strides=[4, 1])

        self.conv4 = tf.keras.layers.Conv2D(256, [32, 1], padding='SAME', activation=tf.nn.relu, dilation_rate=(2,1))
        
        self.maxpool4 = tf.keras.layers.MaxPool2D(pool_size=[4, 1], strides=[4, 1])
        #print(maxpool4)
        self.conv3_1_1 = tf.keras.layers.Conv2D(128, [4, 1], padding='SAME', activation=tf.nn.relu, dilation_rate=(2,1))
        self.conv4_1_1 = tf.keras.layers.Conv2D(128, [4, 1], padding='SAME', activation=tf.nn.relu, dilation_rate=(2,1))
        self.conv5_1_1 = tf.keras.layers.Conv2D(64, [2, 1], padding='SAME', activation=tf.nn.relu, dilation_rate=(2,1))

        self.flatten = tf.keras.layers.Flatten()

        #
        # Bayesian layers
        # #
        self.linear1_1 = tfp.layers.DenseFlipout(128,  activation=tf.nn.relu,seed=self.seed_set(), name='linear1_1')
        self.linear1_2 = tfp.layers.DenseFlipout(128,  activation=tf.nn.relu,seed=self.seed_set(), name='linear1_2')
        self.linear1_3 = tfp.layers.DenseFlipout(128,  activation=tf.nn.relu,seed=self.seed_set(), name='linear1_3')

        self.linear2_1 = tfp.layers.DenseFlipout(128, activation=None,seed=self.seed_set(), name='linear2_1')
        self.linear2_2 = tfp.layers.DenseFlipout(128, activation=None,seed=self.seed_set(), name='linear2_2')
        self.linear2_3 = tfp.layers.DenseFlipout(128, activation=None,seed=self.seed_set(), name='linear2_3')

            
        self.final_1 = tfp.layers.DenseFlipout( 1,  activation=tf.nn.tanh,seed=self.seed_set(), name='final_1') # or softplus
        self.final_2 = tfp.layers.DenseFlipout( 1,  activation=tf.nn.tanh,seed=self.seed_set(), name='final_2') # or softplus
        self.final_3 = tfp.layers.DenseFlipout( 1,  activation=tf.nn.tanh,seed=self.seed_set(), name='final_3')

        
        self.concat = tf.keras.layers.Concatenate(axis=1)

    
    
    def call(self, inputs):
        x = self.reshape1(inputs)
        x = self.conv1(x)
        x= self.maxpool1(x) 
        x= self.conv2(x) 
        x= self.maxpool2(x)
        x=self.conv3(x)
        x=self.maxpool3(x)
        x=self.conv4(x)
        x=self.maxpool4(x)
        x =self.conv3_1_1(x)
        x =self.conv4_1_1(x)
        x =self.conv5_1_1(x)
        
        x1=x
        x2=x
        x3=x

        x1 =self.flatten(x1)
        x2 =self.flatten(x2)
        x3 =self.flatten(x3)
        
        x1=self.linear1_1(x1)
        x2=self.linear1_2(x2)
        x3=self.linear1_3(x3)
        
        x1 = self.linear2_1(x1)
        x2 = self.linear2_2(x2)
        x3 = self.linear2_3(x3)

        x1=self.final_1(x1)
        x2=self.final_2(x2)
        x3=self.final_3(x3)

        outputs = self.concat([x1, x2, x3]) 

        return outputs


class RUN_Model(object):
    def __init__(self,sess_conf,args,model,batch_size, signal_length, num_of_pred_var=3, SEED=None,beta1=1., beta2=1., beta3=1.):
        self.seed_set = SEED
        self.batch_size = batch_size
        self.signal_length = signal_length
        self.num_of_pred_var = num_of_pred_var
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.model = model

        self.sess_conf = sess_conf
        self.vary_var =args.vary_var
        self.fixed_var =args.fixed_var
        self.alpha_KL = args.alpha_KL
        self.test_iter = args.TestIter
        self.Num_monte_carlo = args.num_mc
        self.Restart = args.Re_Start
        self.Input_Lr = args.lr

    
        self.hvd = args.hvd
        self.print_output=args.pout
        self.Run_Testdata =args.TestDataRun
        self.hooks=[]
        self.checkpoint_dir=None
        self.train_logs = args.train_logs
        self.tolerance = args.toler

        
    
    def gen_priordist(std=1):
        def default_multivariate_normal_fn(dtype, shape, name, trainable,add_variable_fn, std=std):
            del name, trainable, add_variable_fn   # Clear unused variable

            dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(std))
            # keep mean as 0, change scales
            batch_ndims = tf.size(dist.batch_shape_tensor())
            return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
        return default_multivariate_normal_fn
    

    def plot_weight_posteriors(self,names, qm_vals, qs_vals, fname):
        self.fig = figure.Figure(figsize=(12, 6))
        self.canvas = backend_agg.FigureCanvasAgg(self.fig)
        
        self.ax = self.fig.add_subplot(1, 2, 1)

        for n, qm in zip(names, qm_vals):
            sns.distplot(qm.flatten(), ax=self.ax, label=n)
        self.ax.set_title("weight means")
        #ax.set_xlim([-1.5, 1.5])
        self.ax.legend()
        
        self.ax = self.fig.add_subplot(1, 2, 2)
        for n, qs in zip(names, qs_vals):
            sns.distplot(qs.flatten(), ax=self.ax)
        self.ax.set_title("weight stddevs")
        #ax.set_xlim([0, 1.])
        
        self.fig.tight_layout()
        self.canvas.print_figure(fname, format="png")
        #print("saved {}".format(fname))
        return

    
    def train(self,dataset, iterations=1000, print_step=50, dropout_rate=0.10):

        # TIMING Start
        Timing_Train = time.time()

        self.input_vector = tf.placeholder(tf.float32, [None, self.signal_length], name='input_layer')
        self.output_vector = tf.placeholder(tf.float32, [None, self.num_of_pred_var], name='ref_layer')
        self.mean = tf.placeholder(tf.float32, [None, self.num_of_pred_var], name='ref_mean')
        self.std = tf.placeholder(tf.float32, [None, self.num_of_pred_var], name='ref_std')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        
        self.dropout_rate = tf.placeholder(tf.float32, [1], name='dropout_rate_init')
        self.if_training = tf.placeholder(tf.bool, [1], name='training_indicator')


        class Dummy():
            pass

        net = Dummy()
        net.plot = Dummy()
        net.plot.RuntimeError = []
        net.plot.RuntimeError_Uncertain=[]
        net.plot.Loss = []

        net.plot.TestRuntimeError = []
        net.plot.TestRuntimeError_Uncertain=[]
        net.plot.TestLoss = []

        net.plot.global_step = []

        net.plot.RunTimeTraining = []
        net.plot.TotalRunTimeTraining = []
        
        
        #
        # Bayesian Loss Defination
        # 
        self.N = self.batch_size
        
        self.outputs = self.model(self.input_vector)

        if self.vary_var:
            self.labels_distribution = tfd.Normal(loc=self.outputs[...,0:6:2],scale=1.e-3+tf.math.softplus(0.05 *self.outputs[...,1:6:2]))
        else:
            self.labels_distribution = tfd.Normal(loc=self.outputs,scale=self.fixed_var*tf.ones(1))

        # Seed input
        self.sample_distribution = self.labels_distribution.sample(seed=self.seed_set())
        #self.sample_distribution = self.labels_distribution.sample()
        
        # Compute the -ELBO as the loss, averaged over the batch size.
        self.neg_log_likelihood = -tf.reduce_mean(input_tensor=self.labels_distribution.log_prob(self.output_vector))

        
        self.KL = sum(self.model.losses)/ tf.cast(self.N,dtype=tf.float32)
        self.elbo_Loss = self.neg_log_likelihood + self.alpha_KL * self.KL

        self.global_step = tf.train.get_or_create_global_step()
        self.get_global_step =tf.train.get_global_step()

        #self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.elbo_Loss,global_step=self.global_step)

        opt = tf.train.AdamOptimizer(self.learning_rate)

        if self.hvd:
            import horovod.tensorflow as hvd
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            #rank = comm.Get_rank() 

            self.num_workers = hvd.size()
            self.opt = hvd.DistributedOptimizer(opt)
        else: 
           self.opt=opt
           self.num_workers = 1
           from mpi4py import MPI
           comm = MPI.COMM_WORLD
        self.train_op = self.opt.minimize(self.elbo_Loss, global_step=self.global_step)


        if self.hvd:
            self.bcast = hvd.BroadcastGlobalVariablesHook(0)
            self.hooks.append(hvd.BroadcastGlobalVariablesHook(0))
            num_steps_hook = tf.train.StopAtStepHook(last_step=iterations // hvd.size())
            #self.hooks.append(saver_hook)
            if hvd.rank() == 0:
                print('Number of iterations Each Rank:{} Total iteration {}'.format(iterations // hvd.size(), iterations))
                self.checkpoint_dir = os.path.join(dataset.final_save_path,self.train_logs)
                ##################################
                # Generate the Test data 
                ###################################
                test_data_batch, test_output_batch = dataset.test_batch(self.batch_size)
                
                # Start a File to write....
                Results_file =  os.path.join(dataset.final_save_path,'Timing_results_'+str(hvd.size())+'.txt')
                if os.path.exists(Results_file):
                    # append the file
                    f = open(Results_file,"a")
                else:
                    f = open(Results_file,"w",1)
                    
            else:
                self.checkpoint_dir = None
        else:
            num_steps_hook = tf.train.StopAtStepHook(last_step=iterations) 
            self.checkpoint_dir = os.path.join(dataset.final_save_path,self.train_logs)

            test_data_batch, test_output_batch = dataset.test_batch(self.batch_size)
            
            # Start a File to write..... 
            Results_file =  os.path.join(dataset.final_save_path,'Timing_results_SingleRank'+'.txt')

            if os.path.exists(Results_file):
                # append the file
                f = open(Results_file,"a")
            else:
                f = open(Results_file,"w",1)
                    
            
            #self.hooks.append(saver_hook)
        
        self.hooks.append(num_steps_hook)

        self.relative_error = tf.identity(tf.reduce_mean(tf.cast(tf.divide(tf.abs(self.output_vector - self.mean), tf.abs(self.output_vector)),
                                                            dtype=tf.float32), axis=0), name="relative_error")
        

        self.relative_error_uncertainty = tf.identity(tf.reduce_mean(tf.cast(tf.divide(self.std, self.output_vector), dtype=tf.float32), axis=0), name="relative_uncertainity")
        


        qmeans = []
        qstds = []
        names = []
        bnn_layers = ['linear1_1','linear1_2', 'linear1_3', 'linear2_1', 'linear2_2', 'linear2_3', 'final_1', 'final_2', 'final_3']
        
        for i, layer in enumerate(bnn_layers):
            try:
                #print(layer)
                q = self.model.get_layer(layer).kernel_posterior
            except AttributeError:
                #print ("I am continuing layer has no attribute")
                continue
            names.append("{}".format(layer))
            qmeans.append(q.mean())
            qstds.append(q.stddev())

        
        MonitoredTiming_Train = time.time()
        

        with tf.train.MonitoredTrainingSession(checkpoint_dir=self.checkpoint_dir,hooks=self.hooks,config=self.sess_conf) as self.sess:
            
            local_step_val=0
            first_global_step = self.sess.run(self.get_global_step)

            self.SNR_FLAG = 0

            #Train Time start 
            start_Train_time = time.time()
            Time_to_train = 0

            Value_Bool = self.sess.should_stop()
            #print(Value_Bool) 

            while not Value_Bool:   
                if self.hvd:
                    global_step_val = self.sess.run(self.global_step)

                    global_step_val_index = global_step_val * hvd.size() # This  is important since the data-set generation step has if-else conditions
                else:
                    global_step_val = self.sess.run(self.global_step)
                    global_step_val_index = global_step_val

                i = local_step_val
                
                ## Setup Adaptive learning_rate
                if i >= 100:
                    if self.hvd:
                        lr = self.Input_Lr * hvd.size()
                    else:
                        lr = self.Input_Lr
                    # if i == 100:
                    #     print ('learning_rate change')
                else:
                    lr = self.Input_Lr
                ##
                
                start_batch = time.time()

                (train_batch_input, train_batch_output) = dataset.train_batch_Flag(self.SNR_FLAG,int(i),self.batch_size)
                
                train_batch_output = train_batch_output[:, 2:] / scale_factor
                train_batch_output[:, 0] = train_batch_output[:, 0] / spin_scale_factor
                
                


                self.sess.run(self.train_op, feed_dict={self.input_vector: train_batch_input,
                                                        self.output_vector: train_batch_output,
                                                        self.dropout_rate: [dropout_rate],
                                                        self.learning_rate: lr,
                                                        self.if_training: [True]})
                
                end_time = time.time()
                time_per_batch = end_time - start_batch
                
                Time_to_train += time_per_batch

                diff_time = end_time - start_Train_time

                if self.hvd:
                    if hvd.rank() ==0:
                        f.write(str(global_step_val) + "," + str(hvd.size()*float(self.batch_size)/time_per_batch) + "," + str(start_batch)+","+str(end_time)+ ","+ str(Time_to_train) +"\n")
                        f.flush()
                else:
                    f.write(str(global_step_val) + "," + str(1.0*float(self.batch_size)/time_per_batch) + "," + str(start_batch)+","+str(end_time)+ ","+ str(Time_to_train) +"\n")
                    f.flush()
         



                net.plot.RunTimeTraining.append(diff_time)
                self.print_output = False
                self.Run_Testdata = False
                self.Check_RelativeError = False

                if self.hvd:
                    Rank = hvd.rank()
                    if hvd.rank()==0:
                        if (i % print_step == 0  and i > 0) : self.print_output=True
                        if (i % self.test_iter == 0) and i > 0: self.Run_Testdata=True
                        if (i % 100 == 0 and i > 0): self.Check_RelativeError=True # check relative error Every 100 local iteration
                else:
                    Rank = 0 # Serial Run
                    if (i % print_step == 0 and i > 0 ):self.print_output=True            
                    if (i % self.test_iter == 0) and i > 0: self.Run_Testdata=True
                    if (i % 100 == 0 and i > 0): self.Check_RelativeError=True # check relative error Every 100 local iteration
                

                if self.Check_RelativeError:   
                    # if (relative_error + un)  - Fixed_relative or (relative_error - un)  - Fixed_relative == Toler
                    # mark flag # send flag to datasetgen

                    num_monte_carlo = self.Num_monte_carlo

                    #Mc sampling is expensive so only do this with large num_mc when we have to i.e for testing
                    rvs = np.asarray([self.sess.run(self.sample_distribution, feed_dict={self.input_vector: train_batch_input}) for _ in range(num_monte_carlo)])
                    mean_mc = np.mean(rvs, axis=0,dtype=np.float32)
                    std_mc = np.std(rvs, axis=0,dtype=np.float32)

                    AvRelError=self.sess.run(self.relative_error,
                                            feed_dict={self.mean:mean_mc,self.learning_rate: lr,
                                                    self.output_vector: train_batch_output,
                                                    self.input_vector: train_batch_input})

                    Uncertainty_on_AvRelError=self.sess.run(self.relative_error_uncertainty,
                                                            feed_dict={self.mean:mean_mc,
                                                                    self.output_vector: train_batch_output,
                                                                    self.input_vector: train_batch_input,
                                                                    self.learning_rate: lr,
                                                                    self.std:std_mc})
                    
                    print('***{},{},{},{},{},{}'.format(self.SNR_FLAG,int(i),global_step_val,AvRelError,Uncertainty_on_AvRelError,Time_to_train))


                    Upper_bound = AvRelError #+Uncertainty_on_AvRelError
                    #Lower_bound = AvRelError-Uncertainty_on_AvRelError
                    Rel_target = np.array([0.09987275, 0.03561291, 0.03175599])
                    #Rel_target = np.array([0.18673794, 0.08693607, 0.09389067])
                    
                    tol = self.tolerance
                    
                    #pred_tol = np.linalg.norm((Rel_target - AvRelError), ord=1 ) -  np.linalg.norm(Uncertainty_on_AvRelError,ord=1) 
                    
                    

                    UP = np.linalg.norm(Upper_bound - Rel_target)
                    #LOW = np.linalg.norm(Lower_bound - Rel_target)
                    
                    print ("UP {}:".format(UP))
                    
                    # if (  UP <= tol or  LOW <= tol):
                    # if( abs(pred_tol) <= tol ):
                    if (UP <= tol):
                        print ('Global_Step_val {} Done with SNR {} RunTime {}'.format(global_step_val_index,self.SNR_FLAG,diff_time))
                        Value_Bool = True
                        comm.bcast(Value_Bool, root=0)
                        print ('Exiting While Loop....')
                        #break
                        #self.Update_SNRFLAG =True
                
                #print("Rank {} Value_Bool {} ".format(hvd.rank(),Value_Bool))
                # TODO: With MPI4PY BroadCasting
                # if hvd.rank() >= 0:
                #     # Check for SNR FLAG
                #     if self.Update_SNRFLAG:
                #         self.SNR_FLAG += 1
                #     self.Update_SNRFLAG = False 

                
                if self.print_output:
                    loss_train, outputs = self.sess.run([self.elbo_Loss, self.outputs], 
                                                                    feed_dict={self.input_vector: train_batch_input, 
                                                                                self.output_vector: train_batch_output, 
                                                                                self.learning_rate: lr,
                                                                                self.dropout_rate: [dropout_rate], 
                                                                                self.if_training: [True]})
                    
                    
                    #print('#{},{},Loss: {}, AvRelError = {}, +/- {}'.format(int(i),global_step_val,loss_train.mean(),AvRelError,Uncertainty_on_AvRelError))
                    #print('#{},{},{}'.format(int(i),global_step_val,loss_train.mean()))
                    
                    
                    
                    # self.qm_vals, self.qs_vals = self.sess.run((qmeans, qstds))

                    # # Store the data for future plotting and visualization
                    # #print ('Evalated qm_vals, qs_vals now storing...!',flush=True)
                    # with open(dataset.final_save_path + "/SavedWeight_Mean_STD_Rank_0_"+str(global_step_val_index), "wb") as out:
                    #     pickle.dump([names,self.qm_vals, self.qs_vals], out)
                    
                    # if HAS_SEABORN:
                    #     #print ('Evalated qm_vals, qs_vals now Plotting & storing...!',flush=True)
                    #     self.fname = dataset.final_save_path + "/SavedPlot_Mean_STD_Rank_0_"+str(global_step_val_index)+'.png'
                    #     self.plot_weight_posteriors(names, self.qm_vals, self.qs_vals, self.fname)

                    
                    net.plot.global_step.append(global_step_val)
                    net.plot.RuntimeError.append(AvRelError)
                    net.plot.RuntimeError_Uncertain.append(Uncertainty_on_AvRelError)
                    net.plot.Loss.append(loss_train.mean())
                    
                ###################################
                #### Update the Local Step
                ###############################
                local_step_val =local_step_val+1

                    
                #####    
                # Testing Every # iteration on Trained model
                #####  
                
                if self.Run_Testdata:

                    Test_loss_train, Test_outputs = self.sess.run([self.elbo_Loss, self.outputs], 
                                                                    feed_dict={self.input_vector: test_data_batch, 
                                                                                self.output_vector: test_output_batch[:, 2:], 
                                                                                self.learning_rate: lr,
                                                                                self.dropout_rate: [dropout_rate], 
                                                                                self.if_training: [False]})
                    # Generate Dataset for 1 SNR and 1 Batch
                    Test_rvs = np.asarray([self.sess.run(self.sample_distribution, feed_dict={self.input_vector: test_data_batch})
                                for _ in range(num_monte_carlo)])
                    Test_mean_mc = np.mean(rvs, axis=0)
                    Test_std_mc = np.std(rvs, axis=0)
                    

                    # Test data Error
                    Test_AvRelError=self.sess.run(self.relative_error,
                                            feed_dict={self.mean:Test_mean_mc,
                                                        self.learning_rate: lr,
                                                    self.output_vector: test_output_batch[:,2:],
                                                    self.input_vector: test_data_batch})

                    Test_Uncertainty_on_AvRelError=self.sess.run(self.relative_error_uncertainty,
                                                            feed_dict={self.mean:Test_mean_mc,
                                                                        self.learning_rate: lr,
                                                                    self.output_vector: test_output_batch[:,2:],
                                                                    self.input_vector: test_data_batch,
                                                                    self.std:Test_std_mc})


                    
                    print('##{},{},{},{},{},'.format(int(i),global_step_val,Test_loss_train.mean(),Test_AvRelError,Test_Uncertainty_on_AvRelError))

                    
                    net.plot.TestRuntimeError.append(Test_AvRelError)
                    net.plot.TestRuntimeError_Uncertain.append(Test_Uncertainty_on_AvRelError)
                    net.plot.TestLoss.append(Test_loss_train.mean())

            
            if self.hvd:
                if hvd.rank()==0:
                    f.close()
            else:
                f.close()
                
            print ("Rank {} Value Bool {}".format(hvd.rank(),Value_Bool))
            loop_end_time = time.time()    
            last_global_step= global_step_val

            # Num steps each worker took (or number of batches each worker saw)
            num_steps = last_global_step - first_global_step
            
            # Elapsed-Time for each worker
            elapsed_time = loop_end_time - start_Train_time
            
            sample_per_sec = (self.num_workers * num_steps * self.batch_size / elapsed_time)

            self.save_output = False
            if self.hvd:
                if hvd.rank() == 0 :
                    print('*'*60)
                    print('Num Steps :{}'.format(num_steps))
                    print('batch_size :{}'.format(self.batch_size))
                    print('Elapsed Time : {}'.format(elapsed_time))
                    print('Num of worker : {}'.format(self.num_workers))
                    print ('sample_per_sec: {}'.format(sample_per_sec))
                    print('*'*60)
                    self.save_output=True
            else:
                print('*'*60)
                print('Num Steps :{}'.format(num_steps))
                print('batch_size :{}'.format(self.batch_size))
                print('Elapsed Time : {}'.format(elapsed_time))
                print('Num of worker : {}'.format(self.num_workers))
                print ('sample_per_sec: {}'.format(sample_per_sec))
                print('*'*60)
                self.save_output=True
                    

            if self.save_output:
                with open(dataset.final_save_path + "/PlotIteration_Train_"+str(Rank), "wb") as out:
                    pickle.dump([net.plot.global_step,net.plot.RuntimeError, net.plot.RuntimeError_Uncertain,net.plot.Loss], out)
                    
                with open(dataset.final_save_path + "/PlotIteration_Test_"+str(Rank), "wb") as out:
                    pickle.dump([net.plot.TestRuntimeError, net.plot.TestRuntimeError_Uncertain,net.plot.TestLoss], out)
                
            # Save Run-Time by each Rank
            with open(dataset.final_save_path + "/PlotTiming_"+str(Rank), "wb") as out:
                pickle.dump([net.plot.RunTimeTraining], out)


        
        Final_endTime = time.time() - MonitoredTiming_Train
        net.plot.TotalRunTimeTraining.append(Final_endTime)
        
        # Save Total Run-Time by each Rank
        with open(dataset.final_save_path + "/PlotTotalTraining_"+str(Rank), "wb") as out:
            pickle.dump([net.plot.TotalRunTimeTraining], out)