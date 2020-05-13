# Data Distributed Training of Bayesian Neural Network (BNN)

The repo consist codes for carrying training of BNN-model on distributed systems. 
The main purpose of the code is to analyze the distibuted training performance of BNN's on High Performace Computing clusters. 
The BNN models are implemented using the Tensorflow-probability libarary. The data distribted training is performed using Horovod.


- **Dependencies**
    + python >= 3.5
    + requirements.txt

- **Dataset:** MNIST

- **Model:** Convolutional and Fully Connected (TFP example implementations) 

- **Files include:**  
  + src/CNN_BNN_Model.py
  + DATA/mnist.npz

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
        
       

- **Example Results**:
    - 


- **Model config:**
  + For other information, please check 'python CNN_BNN_Model.py --help'
  
- **Contact**
  + Himanshu Sharma (himanshu90sharma@gmail.com)



