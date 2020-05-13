# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:37:45 2019

@author: himanshu.sharma
"""
import os
import errno
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import time
from tensorflow import keras
import pickle
from mpi4py import MPI
import tensorflow_probability as tfp
import sys


try:
  import seaborn as sns 
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

from matplotlib import figure 
from matplotlib.backends import backend_agg


tfp_layers = tfp.layers

tf.logging.set_verbosity(tf.logging.INFO)

tfd = tfp.distributions

#tf.enable_eager_execution()
#tf.compat.v1.disable_eager_execution()

def Graph_Info_Writer(dirmake):

  # Dump graph Trainable operations and variables.
  with open( os.path.join(dirmake ,"LayerNames.txt"), 'w') as _out:
      total_parameters = 0
      for variable in tf.trainable_variables():
          this_variable_parameters = np.prod([s for s in variable.shape])
          total_parameters += this_variable_parameters
          _out.write("{}\n".format(variable.name))
      
      _out.close()
          
  # Writing the Name of other operations from the Graph.
  F_write = open(os.path.join(dirmake,'Ops_name_BNN.txt'),'w')
  
  for op in tf.get_default_graph().get_operations():
      F_write.write(str(op.name)+'\n')
  F_write.close()
  
  return


def plot_weight_posteriors(names, qm_vals, qs_vals, fname):

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

def BNN_conv_model(feature_shape, Num_class):
    """2-layer convolution model with second last and 
    last layer as Bayesian 
    """
    
#   Define the Model structure 
    model = tf.keras.Sequential(
        [
        (tf.keras.layers.Reshape(feature_shape)),
        (tfp_layers.Convolution2DFlipout(filters=32,kernel_size=[5, 5],activation=tf.nn.relu,padding="SAME",name="Conv_1")),   
        (tf.keras.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2],padding="SAME",name="Max_1")),    
        (tfp_layers.Convolution2DFlipout(256,kernel_size=[5,5],activation=tf.nn.relu,name="Conv_2")),
        # (tf.keras.layers.MaxPooling2D(pool_size=[2, 2],strides=[2, 2],padding='SAME',name="Max_2")),
        # (tfp_layers.Convolution2DFlipout(64,kernel_size=[5,5],activation=tf.nn.relu,name="Conv_3")),
        # (tf.keras.layers.MaxPooling2D(pool_size=[2, 2],strides=[2, 2],padding='SAME',name="Max_3")),
        # (tfp_layers.Convolution2DFlipout(64,kernel_size=[5,5],activation=tf.nn.relu,padding="SAME",name="Conv_4")),
        # (tf.keras.layers.MaxPooling2D(pool_size=[2, 2],strides=[2, 2],padding='SAME',name="Max_4")),
        # (tfp_layers.Convolution2DFlipout(64,kernel_size=[5,5],activation=tf.nn.relu,padding="SAME",name="Conv_5")),
        # (tf.keras.layers.MaxPooling2D(pool_size=[2, 2],strides=[2, 2],padding='SAME',name="Max_5")),
        # (tfp_layers.Convolution2DFlipout(64,kernel_size=[5,5],activation=tf.nn.relu,padding="SAME",name="Conv_6")),             
        (tf.keras.layers.Flatten()),
        # Bayesian Dense
        (tfp.layers.DenseFlipout(Num_class))
        ]
        )
    return model

def BNN_conv_model_V2(feature_shape, Num_class,filter_size,filter_size2):
    """2-layer convolution model with second last and 
    last layer as Bayesian 
    """
    
#   Define the Model structure 
    model = tf.keras.Sequential(
        [
        (tf.keras.layers.Reshape(feature_shape)),
        (tfp_layers.Convolution2DFlipout(filters=filter_size,kernel_size=[5, 5],activation=tf.nn.relu,padding="SAME",name="Conv_1")),   
        (tf.keras.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2],padding="SAME",name="Max_1")),    
        (tfp_layers.Convolution2DFlipout(filter_size2,kernel_size=[5,5],activation=tf.nn.relu,name="Conv_2")),
        (tf.keras.layers.MaxPooling2D(pool_size=[2, 2],strides=[2, 2],padding='SAME',name="Max_2")),
        (tfp_layers.Convolution2DFlipout(filter_size2,kernel_size=[5,5],activation=tf.nn.relu,name="Conv_3")),
        (tf.keras.layers.MaxPooling2D(pool_size=[2, 2],strides=[2, 2],padding='SAME',name="Max_3")),
        (tfp_layers.Convolution2DFlipout(filter_size2,kernel_size=[5,5],activation=tf.nn.relu,padding="SAME",name="Conv_4")),
        (tf.keras.layers.MaxPooling2D(pool_size=[2, 2],strides=[2, 2],padding='SAME',name="Max_4")),
        (tfp_layers.Convolution2DFlipout(filter_size2,kernel_size=[5,5],activation=tf.nn.relu,padding="SAME",name="Conv_5")),         
        (tf.keras.layers.Flatten()),
        (tfp.layers.DenseFlipout(Num_class))
        ]
        )
    return model

def BNN_conv_model_conv2(feature_shape, Num_class,filter_size,filter_size2):
    """2-layer convolution model with second last and 
    last layer as Bayesian 
    """
    
#   Define the Model structure 
    model = tf.keras.Sequential(
        [
        (tf.keras.layers.Reshape(feature_shape)),
        (tfp_layers.Convolution2DFlipout(filters=filter_size,kernel_size=[5, 5],activation=tf.nn.relu,padding="SAME",name="Conv_1")),   
        (tf.keras.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2],padding="SAME",name="Max_1")),    
        (tfp_layers.Convolution2DFlipout(filter_size2,kernel_size=[5,5],activation=tf.nn.relu,name="Conv_2")),         
        (tf.keras.layers.Flatten()),
        (tfp.layers.DenseFlipout(Num_class))
        ]
        )
    return model

def BNN_FC_model(feature_shape, neurons,Num_class):
    """6-layer Denseflipout model and 
    last layer as simpledense 
    """
#   Define the Model structure 
    model = tf.keras.Sequential([
        (tfp_layers.DenseFlipout(neurons,activation=tf.nn.relu,input_shape=(28*28,) , name="den_1" )),
        (tfp_layers.DenseFlipout(neurons,activation=tf.nn.relu,  name="den_2")),
        (tfp_layers.DenseFlipout(neurons,activation=tf.nn.relu , name="den_3")),
        (tfp_layers.DenseFlipout(neurons,activation=tf.nn.relu, name="den_4")),
        (tfp_layers.DenseFlipout(neurons,activation=tf.nn.relu, name="den_5")),
        (tfp_layers.DenseFlipout(Num_class,name="den_6"))])
    return model

def BNN_FC_model_L3(feature_shape, neurons,Num_class):
    """3-layer Denseflipout model and 
    last layer as simpledense 
    """
#   Define the Model structure 
    model = tf.keras.Sequential([
        (tfp_layers.DenseFlipout(neurons,activation=tf.nn.relu,input_shape=(28*28,), name="den_1" )),
        (tfp_layers.DenseFlipout(neurons,activation=tf.nn.relu,  name="den_2")),
        (tfp_layers.DenseFlipout(Num_class,name="den_3"))])
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
    # Horovod: initialize Horovod.
    hvd.init()
    
    
    class Dummy():
        pass
    

    # TODO: Argument parser
    Num_iter= int(sys.argv[1])
    Model_index= int(sys.argv[2])
    

    print("Total Number of workers",hvd.size())
    print("Rank is:", hvd.rank())
    
    dirmake = "results/result" + "BNN_Test_node" + str(hvd.size())+ "iter"+str(Num_iter)+"Model_"+ str(Model_index)+"/"
    #logdir = os.pathjoin(dirmake,("LOG" + str(hvd.size())+ "/"))
    if MPI.COMM_WORLD.Get_rank() == 0:
        if not os.path.exists(dirmake):
            os.makedirs(dirmake)        
   
    # Load MNIST dataset.
    # (60000,28,28), (60000)
    # (10000,28,28), (10000)
    with np.load('/home/hsharma/WORK/Project_BNN/Theta_data/mnist.npz') as f:
        x_train,y_train = f['x_train'],f['y_train']
        x_test,y_test = f['x_test'],f['y_test']
        
    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
    x_train = np.reshape(x_train, (-1, 784)) / 255.0
    
    x_test = np.reshape(x_test, (-1, 784)) / 255.0
    
    #y_train = tf.one_hot(tf.cast(y_train, tf.int32), 10, 1, 0)
    
    print (y_train.shape)
    # Build model...
    #with tf.name_scope('input'):
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.int32, [None])
    
    K = 10 # number of classes
    feature_shape = [28,28,1]
    # model = BNN_conv_model(feature_shape,K)
    filter_size = 256 # model:12 64 model 13: 128 model:14 256 model-15: 256,64 
    filter_size2 = 94

    # Conv Model
    # model = BNN_conv_model_V2(feature_shape, K,filter_size,filter_size2)
    
    # Model 20 : 256, 14000 
    # model = BNN_conv_model_conv2(feature_shape,K,filter_size,filter_size)

    # FC Model # model-16: 64 # model-17 : 8000-iter,64 neurons # model-18 : 30000-iter,64 neurons 
    # model = BNN_FC_model(feature_shape,filter_size2,K) 

    # 3-FC Model # model-19: 256,14000 # model-21: 94,14000 (70% pruned of original model 256)
    model = BNN_FC_model_L3(feature_shape,filter_size2,K)


    logits = model(image)
    
    # print(logits)
    
    # %% Loss
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    labels_distribution = tfd.Categorical(logits=logits,name='label_dist')
    print('#'*60)
    print(labels_distribution.log_prob(label))
    print('#'*60)
    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(input_tensor=labels_distribution.log_prob(label))
    
    # neg_log_likelihood = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(tf.cast(label, tf.int32), 10, 1, 0), logits=logits))
    
    N = x_train.shape[0]
    kl = sum(model.losses) / N
    
    KLscale=1.0
    elbo_loss = neg_log_likelihood + KLscale * kl
    
    #%%
    #predict, loss = conv_model(image, label, tf.estimator.ModeKeys.TRAIN)
    predictions = tf.argmax(input=logits, axis=1)
    
    train_accuracy, train_accuracy_update_op = tf.metrics.accuracy(labels=label, predictions=predictions)
    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)

    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(elbo_loss, global_step=global_step)

    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step= Num_iter // hvd.size()),

        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': elbo_loss},
                                   every_n_iter=500),
    ]

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    Chk = os.path.join(dirmake,'checkout')
    checkpoint_dir = Chk if hvd.rank() == 0 else None
    training_batch_generator = train_input_generator(x_train,
                                                     y_train, batch_size=100)
    
    ## Creating some variable for runtime writing
    net = Dummy()
    net.plot = Dummy()
    net.Totalruntimeworker = []
    net.plot.RuntimeworkerIter = []
    net.plot.Loss= []
    net.plot.Accuracy= []
    net.plot.Iter_num= []

    #######################################################################
    # Storing mean and standard deviations for the trained weights
    #######################################################################
    qmeans = []
    qstds = []
    names = []
    # Actual_name = []
    for i, layer in enumerate(model.layers):
        try:
            # print(layer)
            q = layer.kernel_posterior
        except AttributeError:
            #print ("I am continuing layer has no attribute")
            continue
        names.append("Layer {}".format(i))
        # Actual_name.append(layer)
        qmeans.append(q.mean())
        qstds.append(q.stddev())    

    Graph_Info_Writer(dirmake)

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        #Train Time start 
        start_Train_time = time.time()
        iter_num = 0
        model.summary()
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            image_, label_ = next(training_batch_generator)

            _, Acc_,Up_opt,loss_ = mon_sess.run([train_op,train_accuracy,train_accuracy_update_op,elbo_loss],feed_dict={image: image_, label: label_})
            

            iter_num = iter_num + 1

            end_time = time.time()
            diff_time = end_time - start_Train_time
            net.plot.RuntimeworkerIter.append(diff_time)
            net.plot.Iter_num.append(iter_num)
            net.plot.Loss.append(loss_)
            net.plot.Accuracy.append(Acc_)
        
            
            
            if (not mon_sess.should_stop()) == True:
                qm_vals, qs_vals = mon_sess.run((qmeans, qstds))

            if hvd.rank() == 0:
                # First worker do the job
                #train_writer = tf.summary.FileWriter(logdir + '/train/{}'.format(epoch), sess.graph)
                print ("Rank-0 Batch RunTime: {:.3f} Acc: {:0.3f} Elbo_loss: {:0.3f} "
                                "Up_opt: {:0.3f}".format(diff_time,Acc_, loss_,Up_opt))
            
                if iter_num % 2000 == 0:
                    if HAS_SEABORN:
                        #print ('Evalated qm_vals, qs_vals now Plotting & storing...!',flush=True)
                        fname = dirmake + "SavedPlot_Mean_STD"+str(iter_num)+'.png'
                        plot_weight_posteriors(names, qm_vals, qs_vals, fname)
           

        # Train Time End
        end_Train_time = time.time()
        diff_trainSess = end_Train_time-start_Train_time
        net.Totalruntimeworker.append(diff_trainSess)
        
        if HAS_SEABORN:
            #print ('Evalated qm_vals, qs_vals now Plotting & storing...!',flush=True)
            fname = dirmake + "SavedPlot_Mean_STD"+str(iter_num)+'.png'
            plot_weight_posteriors(names, qm_vals, qs_vals, fname)


        with open(dirmake + "SavedWeight_Mean_STD_DATA_"+str(iter_num), "wb") as out:
                        pickle.dump([names,qm_vals, qs_vals], out)
        
        # Each Rank dumps results for Training Time  
        with open(dirmake + "TotalRunTime" + str(hvd.rank()), "wb") as out:
            pickle.dump(net.Totalruntimeworker, out)
    

        # Each Rank dumps results for Accu         
        with open(dirmake + "PlotRunTimeIteration" + str(hvd.rank()), "wb") as out:
            pickle.dump([net.plot.RuntimeworkerIter,
                        net.plot.Loss,
                        net.plot.Accuracy,
                        net.plot.Iter_num], out)
        

        # Dump graph Trainableops
        with open( os.path.join(dirmake ,"LayerNames.txt"), 'w') as _out:
            total_parameters = 0
            for variable in tf.trainable_variables():
                this_variable_parameters = np.prod([s for s in variable.shape])
                total_parameters += this_variable_parameters

                _out.write("{}\n".format(variable.name))
            _out.close()
                # _out.write("{} has shape {} and {} total paramters to train.\n".format(
                #     variable.name,
                #     variable.shape,
                #     this_variable_parameters
                # ))
                # _out.write( "Total trainable parameters for this network: {} \n".format(total_parameters))
        
        F_write = open(os.path.join(dirmake,'Ops_name_MNISTBNN.txt'),'w')
        for op in tf.get_default_graph().get_operations():
            F_write.write(str(op.name)+'\n')
        F_write.close()

        # with open( os.path.join(dirmake ,"Model_Summary.txt"), 'w') as _out_1:
        #     _out_1.write(str(model.summary()))
        #     _out_1.flush()
        #     _out_1.close()


if __name__ == "__main__":
    tf.app.run()
