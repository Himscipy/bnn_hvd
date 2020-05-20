# Bayesian Neural Network (BNN) Distributed Training

![](/misc/Repo_Logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The repo consist codes for preforming distributed training of Bayesian Neural Network models at scale using 
High Performance Computing Cluster such as ALCF (Theta). The main purpose of the code is to act as a tutorial for getting 
started with distibuted training of BNN's on High Performace Computing clusters. The BNN's are also used in one of my works for [Gravitational Wave parameters link](https://arxiv.org/abs/1903.01998) where Neural Networks were combined with Bayesian Neural Network Layers. The dataset and the code is available on Theta and restricted to the mmadsp users only. For furthter details about [ADSP](https://www.alcf.anl.gov/science/adsp-allocation-program) contact Argonne ALCF support.

The BNN models are implemented using the Tensorflow-probability libarary. The data distribted training is performed using Horovod.

## **Brief Background on BNN:**
**Bayesian Neutal Networks** is one of approaches used to capture network uncertainity. The uncertainities in Bayesian modeling can be classified under two categories; 

1. Aleatoric uncertainity 
2. Epistemic uncertainity. 

The Aleatoric uncertainity tries to capture noise inherent with the observations/data. The noise in data is associated with sensor measurement noise. Epistemic unceratinity is associated with model parameters, and with increasing the data the uncertainity can be reduced. The Aleoteric uncertainity is further divided into  Homoscedastic and Heteroscedastic.
- **Homoscedastic uncertainty:** uncertainty which stays constant for different inputs, and heteroscedastic uncertainty.  
- **Heteroscedastic uncertainty:** depends on the inputs to the model, with some inputs potentially having more noisy outputs than others. This is particuraly important to avoid model over-confident predictions. 


The Epistemic uncertainty is modelled by putting a prior distribution over the model parameters/weights and compute how these weights varies and converges, which are done in case of Bayesian Neural Networks. While in case of Aleoteric uncertainity are modelled by putting distibutions on the output of the model.
Further, details about the Bayesian Network and Variationa inference for training can be found in the [Jupyter-Notebook](src/VariationalInferenceNotebook.ipynb).

## **Code Dependencies:**
 + python >= 3.5
 + [requirements.txt](requirements.txt)

## **Dataset:** 
 + [MNIST](http://yann.lecun.com/exdb/mnist/) hand-written digit dataset sample images below.

     ![](misc/MNIST_DataSamples.png)
 
 + [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
      ![](misc/CIFAR10_Sample.png)
  

## **Models:** 
  + Bayesian Neural Network with Flipout Fully Connected Layer.('BNN_conv_flip')
  + Bayesian Neural Network with Non-Flipout Fully Connected Layer.('BNN_conv_nonflip')
  + Bayesian Neural Network with Flipout Convolutional Layers.('BNN_FC_flip')
  + Bayesian Neural Network with Non-Flipout Convolutional Layers.('BNN_FC_nonflip)
  + Bayesian Neural Network with Flipout Convolutional Layers (3-VGG Blocks) for CIFAR-10 data.('CIFAR10_BNN_model')
  + Convolutional Neural Network ('CNN_Conv')
  + Fully Connected Neural Network ('CNN_FC')


## **How to run the code:**
   - On the local machine Running:  
        + `horovodrun -n 2 -H localhost:2 python CNN_BNN_Model.py --flagfile=config_file.cfg`

   - ALCF high performance Computing Cluster (Theta): 
   ![](misc/Theta1600x900.jpg) 
         
        ```
        PPN=1 # 32,16,8 MPIRank Per Node (Process Per Node)
        NUM_THDS=128

        aprun -n $((${COBALT_PARTSIZE} * ${PPN})) -N ${PPN} -cc depth -j 2 -d ${NUM_THDS} \
            -e OMP_NUM_THREADS=${NUM_THDS} -b python <path to the code>/CNN_BNN_Model.py \
            --flagfile=config_file.cfg
        ```

- The submission [script](/src/RUN_ScriptBNN/run_script.sh) is provided in the repository.

- Running the job with [Balsam](https://www.alcf.anl.gov/support-center/theta/balsam) (Theta):
    + Look the detail [README.md](BalsamJobCreation/Readme.md)
    
- For other information about the configuration of running the code, use help function as follows;  
`python CNN_BNN_Model.py --help`

        

- **Example Results:**  
    - The comparison of the BNN and CNN time to train with increasing number of nodes are shown in the Fig-1 shown below.
    ![](misc/Demo_Timing_PlotBNN_CNNComparison.png)
    
    - The comparison of the Speed-Up between the BNN and CNN using the training time can be also compared in Fig-2 shown below.   
    ![](misc/Demo_SpeedUp_PlotBNN_CNNComparison.png)

    - The training of the Bayesian Network is to find optimal distribution of the training parameters which done using the technique of Variational Inference(VI). As the training iteration progress the weights posterior converges. An example is shown below with the weights initialized with the Gaussian prior in Fig-3.
    ![](misc/Repo_Weights_Updates.png)
    
    - As the model is trained once and the posteriors for the weights are converged. The model is used for performing inference. The inference is perfomed by running the model over and over again (Monte-Carlo iterations). The output of the model returns the prediction distribution as shown below for MC iterations of 300 and with a BNN Fully Connected model.
    ![](misc/Repo_DemoResults_FC_layer_combine.png)


- **Research Articles:**
    **Papers related to Bayesian Neural Networks:**
    + [Probabilistic Backpropagation for ScalableLearning of Bayesian Neural Networks](http://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf) 
    + [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424)
    + [Practical Variational Inference for Neural Networks](https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks)
    + [A Comprehensive guide to Bayesian Convolutional NeuralNetwork with Variational Inference](https://arxiv.org/pdf/1901.02731.pdf)
    + [Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches](https://arxiv.org/abs/1803.04386)
    
    **Papers for Gravitational Bayesian Model:**
     + [Deterministic and Bayesian Neural Networks for Low-latency Gravitational Wave Parameter Estimation of Binary Black Hole Mergers](https://arxiv.org/abs/1903.01998)


- **Additional Resources**:
    + [Tensorflow Probalbility Examples](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples)



- **Contact**
  + Himanshu Sharma (himanshu90sharma@gmail.com)
  + Elise Jennings 
  
- **Ackowledegment** 

    ![](misc/ANL_RGB-01.png) 

    This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357. This research was funded in part and used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357. This work describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the work do not necessarily represent the views of the U.S. DOE or the United States Government. Declaration of Interests - None.

