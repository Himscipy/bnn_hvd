# Data Distributed Training of Bayesian Neural Network (BNN)

The repo consist codes for carrying training of BNN-model on distributed systems. 
The main purpose of the code is to analyze the distibuted training performance of BNN's on High Performace Computing clusters. 
The BNN models are implemented using the Tensorflow-probability libarary. The data distribted training is performed using Horovod.


- **Dependencies**
    + python >= 3.5
    + tensorflow >= 1.14
    + tensorflow-probability >= 0.7
    + Horovod >= 0.18.2
    + Seaborn >= 0.10.0

- **Dataset:** MNIST, CIFAR10

- **Model:** Convolutional and Fully Connected  (TFP example implementations) 

- **Files include:**  
  + model/bayesian_vgg.py
  + model/bayesian_resnet.py
  + model/utils.py
  + TFP_CIFAR_10

- **How to run the code :**
   - Local Running:  
        + `horovodrun -n 2 localhost:2 python CIFAR_BNN_CNN.py --flagfile=config_file.cfg`

    - ALCF high performance Computing Cluster (Theta) Running:  
        + `PPN=1 # 32,16,8 MPIRank Per Node (Process Per Node)`  
        + `NUM_THDS=128`  
        ```
        aprun -n $((${COBALT_PARTSIZE} * ${PPN})) -N ${PPN} -cc depth -j 2 -d ${NUM_THDS} \
            -e OMP_NUM_THREADS=${NUM_THDS} -b python <path to the code>/CIFAR_CNN_BNN.py \
            --flagfile=config_file.cfg
        ```

- **Model config:**
  + For other information, please check 'python CIFAR_BNN_CNN.py --help'
  
- **Contact**
  + Himanshu Sharma (himanshu90sharma@gmail.com)



