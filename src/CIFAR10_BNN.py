# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:37:45 2019

@author: himanshu.sharma
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.simplefilter(action="ignore")

import os
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
from utils import Pre_Post_Process,Model_CNN_BNN,Data_API

try:
  import seaborn as sns 
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False


tfp_layers = tfp.layers

tf.logging.set_verbosity(tf.logging.INFO)

tfd = tfp.distributions

IMAGE_SHAPE = [32, 32, 3]

flags.DEFINE_float("learning_rate",default=0.0001,help="Initial learning rate.")
flags.DEFINE_boolean("subtract_pixel_mean",default=True,help="Boolean for normalizing the images")
flags.DEFINE_integer("epochs", default= 700,help="Number of epochs to train for.")
flags.DEFINE_integer("batch_size", default=128, help="Batch size. #16,32,128")
flags.DEFINE_string("model_dir",default=os.path.join('./results',"bayesian_neural_network/"),help="Directory to put the model's fit.")
flags.DEFINE_integer("num_monte_carlo",default=80,help="Network draws to compute predictive probabilities.")
flags.DEFINE_integer("print_step", default=10,help=" Frequency to perform printing")
flags.DEFINE_integer("iterations", default=100,help=" Number of iterations for debugging purpose only")
flags.DEFINE_string("DATA_PATH",default="/projects/datascience/hsharma/bnn_horovod/TFP_CIFAR10/RunScript/cifar-10-batches-py",help="Path of the data directory")
flags.DEFINE_boolean("USE_EPOCH",default="False",help="If set true use the specified epochs to")
flags.DEFINE_integer('num_intra', default=128,help='num_intra')
flags.DEFINE_integer('num_inter', default=1,help='num_inter')
flags.DEFINE_integer('kmp_blocktime', default=0,help='KMP BLOCKTIME')
flags.DEFINE_string('kmp_affinity', default='granularity=fine,verbose,compact,1,0',help='KMP AFFINITY')
flags.DEFINE_bool('verbose',default=False,help='Verbosre to print extra details about the model')
flags.DEFINE_bool('plotting',default=False,help='Plot the layer posterior on/off switch model')
flags.DEFINE_float('kl_regularizer',default=1,help='Value betweeen (0,1]')
flags.DEFINE_integer('eval_freq',default=100,help='Use the eval frequency to do the MC iteration')
flags.DEFINE_string('DATA_NAME',default='CIFAR-10',help='Set the data used for the study (Available: CIFAR-10,CIFAR-100)')


FLAGS = flags.FLAGS





def main(argv):
    tf.reset_default_graph()
    hvd.init()
    
    DataObj = Data_API(FLAGS)
    
    PrePostObj = Pre_Post_Process(FLAGS)
    
    config = PrePostObj.create_config_proto()

    Seed_set = PrePostObj.Setup_Seed(hvd.rank())


    class Dummy():
        pass

    
    del argv  # unused

    if hvd.rank() == 0:
        print("Total Number of workers",hvd.size())
        data_dir = os.path.join(FLAGS.model_dir, 'data')

        if tf.io.gfile.exists(FLAGS.model_dir) and tf.io.gfile.exists(data_dir):
            tf.compat.v1.logging.warning(
                "Exists: log directory at {} for restart".format(FLAGS.model_dir))
        else:
            tf.compat.v1.logging.warning(
                "Warning: Creating log directory at {}".format(FLAGS.model_dir))
            tf.io.gfile.makedirs(FLAGS.model_dir)
            tf.io.gfile.makedirs(data_dir)
    else:
        None

    (x_train, y_train), (x_test, y_test) = DataObj.load_data(hvd.rank())

    K = 10

    ModelObj = Model_CNN_BNN(IMAGE_SHAPE,K)

    if FLAGS.verbose:
         if hvd.rank() == 0:
             print ('Label size',y_train.shape)
             print ('Traint data size',x_train.shape)
    
    with tf.name_scope('input'):
        images = tf.placeholder(tf.float32, [None, *IMAGE_SHAPE], name='image')
        labels = tf.placeholder(tf.int32, [None], name='label')
    

    if FLAGS.USE_EPOCH:
        training_steps = int(round(FLAGS.epochs * (len(x_train) / FLAGS.batch_size)))
    else:
        training_steps = FLAGS.iterations # 100 # Debugging.... only.!



    if hvd.rank() == 0:
        with open( os.path.join(data_dir + "Parsed_Arg"+".log"), 'w') as _out:
            _out.write(str(FLAGS.flag_values_dict()))
            _out.flush()
            _out.close()
        print("Training iterations",training_steps)
        print ('*'*60)
        print (FLAGS.flag_values_dict(),flush=True)
        print ('*'*60)

    
    model = ModelObj.CIFAR10_BNN_model()
    logits = model(images)

    labels_distribution = tfd.Categorical(logits=logits)
    

    sample_label = labels_distribution.sample(FLAGS.num_monte_carlo)
    
    # Perform KL annealing. The optimal number of annealing steps
    # depends on the dataset and architecture.
    
    t = tf.Variable(0.0,use_resource=True)
    
    kl_regularizer = FLAGS.kl_regularizer
    
    # Compute the -ELBO as the loss. The kl term is annealed from 0 to 1 over
    # the epochs specified by the kl_annealing flag.
    log_likelihood = labels_distribution.log_prob(labels)
    neg_log_likelihood = -tf.reduce_mean(input_tensor=log_likelihood)
    kl = sum(model.losses) / len(x_train) * tf.minimum(1.0, kl_regularizer)

    loss = neg_log_likelihood + kl
        

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy
    # predictions.
    predictions = tf.argmax(input=logits, axis=1)


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
    
    with tf.name_scope("train"):
        train_accuracy, train_accuracy_update_op = tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions)

        opt = tf.train.AdamOptimizer(FLAGS.learning_rate * hvd.size())
        
        # Horovod: add Horovod Distributed Optimizer.
        opt = hvd.DistributedOptimizer(opt)
       
        global_step = tf.train.get_or_create_global_step()
        
        train_op = opt.minimize(loss, global_step=global_step)

        update_step_op = tf.assign(t, t + 1)
    
    with tf.name_scope("valid"):
        valid_accuracy, valid_accuracy_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels))
        
        Def_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    hooks = [
            # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
            # from rank 0 to all other processes. This is necessary to ensure consistent
            # initialization of all workers when training is started with random weights
            # or restored from a checkpoint.
            hvd.BroadcastGlobalVariablesHook(0),

            # Horovod: adjust number of steps based on number of CPUs.
            tf.train.StopAtStepHook(last_step= training_steps // hvd.size()),

        ]

    class Dummy():
        pass

    ## Creating some variable for runtime writing
    net = Dummy()
    net.plot = Dummy()
    net.Totalruntimeworker = []
    net.plot.RuntimeworkerIter = []
    net.plot.Loss= []
    net.plot.Accuracy= []
    net.plot.Iter_num= []


    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    checkpoint_dir = os.path.join(FLAGS.model_dir,'train_logs') if hvd.rank() == 0 else None
 
    if hvd.rank() == 0:
        Results_file =  os.path.join(data_dir,('Timing_results_{}_Ranks.txt'.format(hvd.size())) )
        Results_Train_file =  os.path.join(data_dir,('Result_Train_{}_Ranks.csv'.format(hvd.size())) )
        
        if os.path.exists(Results_file):
            # append the file
            f = open(Results_file,"a")
            f_1 = open(Results_Train_file,"a")
        else:
            f = open(Results_file,"w",1)
            f_1 = open(Results_Train_file,"w")
            



    stream_vars_valid = [ v for v in tf.local_variables() if "valid/" in v.name ]
    reset_valid_op = tf.variables_initializer(stream_vars_valid)

 
    training_batch_generator = DataObj.train_input_generator(x_train, y_train, batch_size=FLAGS.batch_size)
    test_batch_generator = DataObj.train_input_generator(x_test, y_test, batch_size=1)


    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,hooks=hooks,config=config) as sess:

        # Uncomment to see model Parameters & Graph
        if FLAGS.verbose:
            if hvd.rank() == 0:
                print(model.summary(),flush=True)

        # Run the training loop
        # Time start
        start_Train_time = time.time()
        Time_to_train = 0
        batch_process = 0

        while not sess.should_stop():

            Image , Label = next(training_batch_generator)
            step = sess.run(global_step)
    
            start_batch_time = time.time()
            
            _ = sess.run([train_op,train_accuracy_update_op,update_step_op],feed_dict={images: Image,labels:Label})
        
            end_batch_time = time.time()

            batch_process += 1
            Time_per_batch = end_batch_time - start_batch_time
            Time_to_train += Time_per_batch
            

            if hvd.rank() ==0: 
                f.write(str(step) + "," + str(hvd.size()*float(FLAGS.batch_size)/Time_per_batch) + "," + str(Time_per_batch)+"," + str(start_batch_time)+","+str(end_batch_time)+ ","+ str(Time_to_train) +"\n")
                f.flush()


            

            # Manually print the frequency
            if step % FLAGS.print_step == 0 and (not sess.should_stop()) == True  :    
                    
                loss_value, accuracy_value, kl_value = sess.run([loss, train_accuracy, kl], feed_dict={images: Image,labels:Label})
                    

                net.plot.Iter_num.append(step)
                net.plot.Loss.append(loss_value)
                net.plot.Accuracy.append(accuracy_value)



                Write = str(step) +','+ str(loss_value)+','+ str(accuracy_value)+','+ str(kl_value)+','+str(Time_to_train)+"\n"
                out_print = "##Step: {:>3d} Loss: {:.3f} Avg_Accuracy: {:.3f} KL: {:.3f}  Time: {:.3f}".format(step, loss_value, accuracy_value, kl_value,Time_to_train)
                

                if hvd.rank() == 0:
                    print(out_print,flush=True)
                    f_1.write(Write)
                    f_1.flush()

                
                qm_vals, qs_vals = sess.run((qmeans, qstds))
                
                
                if hvd.rank() == 0:                
                    
                    #  Store the data for future plotting and visualization
                    # Log_print('Evalated qm_vals, qs_vals now storing...!',hvd.rank())  
                    with open( os.path.join(data_dir , ("SavedWeight_Mean_STD_DATA_{}".format(step)) ), "wb") as out:
                        pickle.dump([names,qm_vals, qs_vals], out)
                
                    if FLAGS.plotting:

                        if HAS_SEABORN:
                            fname = os.path.join(data_dir , ("SavedPlot_Mean_STD_{}.png".format(step)) )
                            PrePostObj.plot_weight_posteriors(names, qm_vals, qs_vals, fname)
    

        end_Train_time = time.time()
        diff_trainSess = end_Train_time-start_Train_time

        with open( os.path.join( FLAGS.model_dir , ("data/PlotRunTimeIteration_Rank_{}".format(hvd.rank()) ) ) , "wb") as out:
            pickle.dump([net.plot.Iter_num,net.plot.Loss,net.plot.Accuracy], out)
    
        if hvd.rank()==0:
            f.close()
            f_1.close()



if __name__ == "__main__":
    tf.app.run()
    