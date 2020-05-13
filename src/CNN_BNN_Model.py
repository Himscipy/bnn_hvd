# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:37:45 2019

@author: himanshu.sharma
"""
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

flags.DEFINE_string("data_dir",default='/home/hsharma/WORK/Project_BNN/bnn_hvd/DATA/mnist.npz',
                    help="Directory where data is stored (if using real data).")
flags.DEFINE_string("model_data",default='./results/',help='Define the data directory')
flags.DEFINE_integer("batch_size",default=64,help='Batch size')
flags.DEFINE_integer("iteration",default=8000,help='Iterations to train for')
flags.DEFINE_bool('compression',default=False,help='Compression of grad to fp16')
flags.DEFINE_bool('verbose',default=False,help='Verbose for printing')
flags.DEFINE_string('cnnConv',default='CNN_conv',help='CNN with Convolutional Layers')
flags.DEFINE_string('bnnConv',default='BNN_conv_flip',help='FC and Convolutional Layers type')
flags.DEFINE_integer("num_intra",default=128,help='Intra Threading')
flags.DEFINE_integer("num_inter",default=1,help='Inter threading')
flags.DEFINE_integer("kmp_blocktime",default=0,help='KMP_BLOCKTIME setting')
flags.DEFINE_string("kmp_affinity",default='granularity=fine,verbose,compact,1,0',help='granularity setting')
flags.DEFINE_string("Model_type",default='BNN', help="Select the type of model options(BNN,CNN)")


FLAGS = flags.FLAGS  

def create_config_proto():
   '''HS: TF config setup'''
   config = tf.ConfigProto()
   config.intra_op_parallelism_threads = FLAGS.num_intra
   config.inter_op_parallelism_threads = FLAGS.num_inter
   config.allow_soft_placement         = True
   os.environ['KMP_BLOCKTIME'] = str(FLAGS.kmp_blocktime)
   os.environ['KMP_AFFINITY'] = FLAGS.kmp_affinity
   return config



def BNN_conv_model(feature_shape, Num_class):
    """1-layer convolutionflipout model  and 
    last layer as dense
    """
    
#   Define the Model structure 
    model = tf.keras.Sequential([
        (tf.keras.layers.Reshape(feature_shape)),
        (tfp_layers.Convolution2DFlipout(64,kernel_size=[5,5],activation=tf.nn.relu,padding='SAME',input_shape=feature_shape)), 
        (tf.keras.layers.MaxPooling2D(pool_size=[2, 2],strides=[2, 2],padding='SAME')),
        (tf.keras.layers.Flatten()),
        (tf.keras.layers.Dense(Num_class))])
    return model


def BNN_FC_model(feature_shape, Num_class):
    """1-layer Denseflipout model and 
    last layer as simpledense 
    """
#   Define the Model structure 
    model = tf.keras.Sequential([
        (tfp_layers.DenseFlipout(64,activation=tf.nn.relu,input_shape=(28*28,))),
        (tf.keras.layers.Dense(Num_class))])
    return model

def BNN_conv_model_nonFlip(feature_shape, Num_class):
    """1-layer convolutionflipout model  and 
    last layer as dense
    """
    
#   Define the Model structure 
    model = tf.keras.Sequential([
        (tf.keras.layers.Reshape(feature_shape)),
        (tfp_layers.Convolution2DReparameterization(64,kernel_size=[5,5],activation=tf.nn.relu,padding='SAME')), 
        (tf.keras.layers.MaxPooling2D(pool_size=[2, 2],strides=[2, 2],padding='SAME')),
        (tf.keras.layers.Flatten()),
        (tf.keras.layers.Dense(Num_class))])
    return model


def BNN_FC_model_nonFlip(feature_shape, Num_class):
    """1-layer Denseflipout model and 
    last layer as simpledense 
    """
#   Define the Model structure 
    model = tf.keras.Sequential([
        (tfp_layers.DenseReparameterization(64,activation=tf.nn.relu,input_shape=(28*28,))),
        (tf.keras.layers.Dense(Num_class))])
    return model



def CNN_conv_model(feature_shape, Num_class):
    """1-layer convolution model with  and 
    last layer as simple dense
    """
#   Define the Model structure 
    model = tf.keras.Sequential([
        (tf.keras.layers.Reshape(feature_shape)),
        (tf.keras.layers.Conv2D(filters=64,kernel_size=[5, 5],
                                activation=tf.nn.relu,
                                padding="SAME",
                                )),   
        (tf.keras.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2],padding="SAME")),    
        (tf.keras.layers.Flatten()),
        (tf.keras.layers.Dense(Num_class))])
    return model

def CNN_FC_model(feature_shape, Num_class):
    """2-layer convolution model with second last and 
    last layer as Bayesian 
    """
#   Define the Model structure 
    model = tf.keras.Sequential([
        (tf.keras.layers.Dense(64,activation=tf.nn.relu,input_shape=(28*28,))),
        (tf.keras.layers.Dense(Num_class))])
    return model



def train_input_generator(x_train, y_train, batch_size=64):
    assert len(x_train) == len(y_train)
    while True:
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size


def main(_):
    tf.reset_default_graph()

    config = create_config_proto()

    # Horovod: initialize Horovod.
    hvd.init()
    
    
    class Dummy():
        pass
    

    if hvd.rank() == 0:
        print ('*'*60)
        print (FLAGS.flag_values_dict())
        print ('*'*60)

    
    Num_iter= FLAGS.iteration 

    print("Total Number of workers",hvd.size())
    print("Rank is:", hvd.rank())
    
    if FLAGS.Model_type == 'CNN':
        dirname = "Result"+"_"+str(FLAGS.Model_type)+"_ConvFlag_"+str(FLAGS.cnnConv)
    else:
        dirname = "Result"+"_"+str(FLAGS.Model_type)+"_ConvFlag_"+str(FLAGS.bnnConv)

    dirmake = os.path.join(FLAGS.model_data,dirname)

    #logdir = os.pathjoin(dirmake,("LOG" + str(hvd.size())+ "/"))
    if hvd.rank() == 0:
        if not os.path.exists(dirmake):
            os.makedirs(dirmake)        
   
    # Load MNIST dataset.
    # (60000,28,28), (60000)
    # (10000,28,28), (10000)
    with np.load(FLAGS.data_dir) as f:
        x_train,y_train = f['x_train'],f['y_train']
        x_test,y_test = f['x_test'],f['y_test']
        
    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
    x_train = np.reshape(x_train, (-1, 784)) / 255.0
    
    x_test = np.reshape(x_test, (-1, 784)) / 255.0
    
    print ('Number of Samples: {}'.format(y_train.shape))
    
    # Build model...
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.int32, [None])
    
    print(tf.shape(label))

    K = 10 # number of classes
    feature_shape = [28,28,1]

    if FLAGS.Model_type == 'BNN':
        
        if FLAGS.bnnConv == 'BNN_conv_flip':
            model = BNN_conv_model(feature_shape,K)
        elif FLAGS.bnnConv == 'BNN_FC_flip':
            model = BNN_FC_model(feature_shape,K)
        elif FLAGS.bnnConv == 'BNN_conv_nonflip':
            model = BNN_conv_model_nonFlip(feature_shape,K)
        elif FLAGS.bnnConv == 'BNN_FC_nonflip':
            model = BNN_FC_model_nonFlip(feature_shape,K)

        logits = model(image)
        
        # %% Loss
        # Convert the target to a one-hot tensor of shape (batch_size, 10) and
        # with a on-value of 1 for each one-hot vector of length 10.
        labels_distribution = tfd.Categorical(logits=logits,name='label_dist')
        
        
        # Compute the -ELBO as the loss, averaged over the batch size.
        neg_log_likelihood = -tf.reduce_mean(input_tensor=labels_distribution.log_prob(label))
        
        N = x_train.shape[0]
        kl = sum(model.losses) / N
        
        KLscale=1.0
        Loss_ = neg_log_likelihood + KLscale * kl
        
                
        predictions = tf.argmax(input=logits, axis=1)
        
        train_accuracy, train_accuracy_update_op = tf.metrics.accuracy(labels=label, predictions=predictions)
        # Horovod: adjust learning rate based on number of GPUs.
        opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())

        if FLAGS.compression:
            # Horovod: add Horovod Distributed Optimizer.
            opt = hvd.DistributedOptimizer(opt,compression=hvd.Compression.fp16)
        else:
            # Horovod: add Horovod Distributed Optimizer.
            opt = hvd.DistributedOptimizer(opt)
            
        global_step = tf.train.get_or_create_global_step()
        train_op = opt.minimize(Loss_, global_step=global_step)

    elif FLAGS.Model_type == 'CNN':
        if FLAGS.cnnConv == 'CNN_conv':
            model = CNN_conv_model(feature_shape,K)
        elif FLAGS.cnnConv ==  'CNN_FC':
            model = CNN_FC_model(feature_shape,K)
        
        logits = model(image)
    
        # %% Loss
        # Convert the target to a one-hot tensor of shape (batch_size, 10) and
        # with a on-value of 1 for each one-hot vector of length 10.
        neg_log_likelihood = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(tf.cast(label, tf.int32), 10, 1, 0), logits=logits))
        
        Loss_ = neg_log_likelihood 

        #%%
        #predict, loss = conv_model(image, label, tf.estimator.ModeKeys.TRAIN)
        predictions = tf.argmax(input=logits, axis=1)
        
        train_accuracy, train_accuracy_update_op = tf.metrics.accuracy(labels=label, predictions=predictions)
        # Horovod: adjust learning rate based on number of GPUs.
        opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())


        if FLAGS.compression:
            # Horovod: add Horovod Distributed Optimizer.
            opt = hvd.DistributedOptimizer(opt,compression=hvd.Compression.fp16)
        else:
            # Horovod: add Horovod Distributed Optimizer.
            opt = hvd.DistributedOptimizer(opt)

        # Horovod: add Horovod Distributed Optimizer.
        # opt = hvd.DistributedOptimizer(opt)

        global_step = tf.train.get_or_create_global_step()
        train_op = opt.minimize(Loss_, global_step=global_step)
    else:
        print ('Not a valid model...!')
        exit()
            
    
    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step= Num_iter // hvd.size()),

        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': Loss_},every_n_iter=500),
    ]

    # # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    checkpoint_dir = os.path.join(dirmake,'checkpoints') if hvd.rank() == 0 else None
    training_batch_generator = train_input_generator(x_train,y_train, batch_size=FLAGS.batch_size)
    
    if hvd.rank() == 0:
        Results_file =  os.path.join(dirmake,'Timing_results_'+str(hvd.size())+'.txt')
        
        if os.path.exists(Results_file):
            # append the file
            f = open(Results_file,"a")
        else:
            f = open(Results_file,"w",1)
    
    ## Creating some variable for runtime writing
    net = Dummy()
    net.plot = Dummy()
    net.plot.Totalruntimeworker = []
    net.plot.Loss= []
    net.plot.Accuracy= []
    net.plot.Iter_num= []

    if hvd.rank() == 0 :
        if FLAGS.verbose:
            with open( dirmake + "/training.log", 'w') as _out:
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
                
   
    
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        #Train Time start 
        start_Train_time = time.time()
        iter_num = 0

        # Uncomment to see model Parameters & Graph
        if FLAGS.verbose:
            if hvd.rank() == 0:
                print(model.summary(),flush=True)
        
        while not mon_sess.should_stop():
            # Run a training step synchronously.
    
            image_, label_ = next(training_batch_generator)

            start_batch_time = time.time()
    
            _, Acc_,Up_opt,loss_= mon_sess.run([train_op,train_accuracy,
                                                        train_accuracy_update_op,Loss_],
                                                        feed_dict={image: image_, label: label_})
            
            end_batch_time = time.time()

            net.plot.Loss.append(loss_)
            net.plot.Accuracy.append(Acc_)
            net.plot.Iter_num.append(iter_num)

            iter_num = iter_num + 1

            diff_time = end_batch_time - start_batch_time
            samp_sec = ( hvd.size()* float(FLAGS.batch_size)/diff_time )

           

            if hvd.rank() == 0:
                f.write("{},{},{},{},{} \n".format(iter_num,diff_time,samp_sec,start_batch_time, end_batch_time ) )
                f.flush()

                if iter_num % 10 == 0 and (not mon_sess.should_stop()) == True:
                    print("Iter:{},Acc:{},Loss:{} \n".format(iter_num,Acc_,loss_) )
            
               
                    
        # Train Time End
        if hvd.rank() == 0:
            f.close()

        end_Train_time = time.time()
        diff_trainSess = end_Train_time - start_Train_time

        net.plot.Totalruntimeworker.append(diff_trainSess)
        
    

        # # Each Rank dumps results for Accu         
        with open( os.path.join(dirmake , ("PlotRunData_Rank_" + str(hvd.rank())) ), "wb") as out:
            pickle.dump([net.plot.Totalruntimeworker,
                        net.plot.Loss,
                        net.plot.Accuracy,
                        net.plot.Iter_num], out)

if __name__ == "__main__":
    tf.app.run()
