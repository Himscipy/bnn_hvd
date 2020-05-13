# Data Distributed Training of Bayesian Neural Network (BNN)

The repo consist codes for preforming distributed training of Bayesian Neural Network models at scale using 
High Performance Computing Cluster such as ALCF (Theta). 
The main purpose of the code is to analyze the distibuted training performance of BNN's on High Performace Computing clusters. 
The BNN models are implemented using the Tensorflow-probability libarary. The data distribted training is performed using Horovod.

- **Dependencies**
    + python >= 3.5
    + requirements.txt

- **Dataset:** 
    + [MNIST] hand-written digit dataset.
    + [CIFAR-10] 

- **Models:** 
    + Bayesian Neural Network with Flipout Fully Connected Layer.('BNN_conv_flip')
    + Bayesian Neural Network with Non-Flipout Fully Connected Layer.('BNN_conv_nonflip')
    + Bayesian Neural Network with Flipout Convolutional Layers.('BNN_FC_flip')
    + Bayesian Neural Network with Non-Flipout Convolutional Layers.('BNN_FC_nonflip)
    
    + Convolutional Neural Network 
    + Fully Connected Neural Network 

- **Files included:**  
  + src/CNN_BNN_Model.py
  + DATA/mnist.npz

- **Model config:**
  + For other information, please check 'python CNN_BNN_Model.py --help'

- **How to run the code :**
   - On the local machine Running:  
        + `horovodrun -n 2 -H localhost:2 python CNN_BNN_Model.py --flagfile=config_file.cfg`

   - ALCF high performance Computing Cluster (Theta) Running:  
         
        ```
        PPN=1 # 32,16,8 MPIRank Per Node (Process Per Node)
        NUM_THDS=128

        aprun -n $((${COBALT_PARTSIZE} * ${PPN})) -N ${PPN} -cc depth -j 2 -d ${NUM_THDS} \
            -e OMP_NUM_THREADS=${NUM_THDS} -b python <path to the code>/CNN_BNN_Model.py \
            --flagfile=config_file.cfg
        ```
    - Running the job with Balsam (Theta):
        
       

- **Example Results:**
    - 

- **Research papers related to Bayesian Neural Networks:**
    + [Probabilistic Backpropagation for ScalableLearning of Bayesian Neural Networks](http://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf) 
    + [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424)
    + [Practical Variational Inference for Neural Networks](https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks)
    + [A Comprehensive guide to Bayesian Convolutional NeuralNetwork with Variational Inference](https://arxiv.org/pdf/1901.02731.pdf)
    + [Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches](https://arxiv.org/abs/1803.04386)
  
- **Contact**
  + Himanshu Sharma (himanshu90sharma@gmail.com)

- **Ackowledegment**  
This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357. This research was funded in part and used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357. This work describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the work do not necessarily represent the views of the U.S. DOE or the United States Government. Declaration of Interests - None. 

