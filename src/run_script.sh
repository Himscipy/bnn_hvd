#!/bin/bash
#COBALT -t 1:00:00
#COBALT -n 2
#COBALT -A datascience
#COBALT -q debug-flat-quad 


# module load datascience/tensorflow-1.14
# module load datascience/horovod-0.18.1
# module load datascience/h5py-2.9.0
module load /projects/datascience/hsharma/bnn_hvd/src/GW_BNN/run_scripts/tensorflow-1.14-HVD

export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:$PYTHONPATH

which python

PPN=1
NUM_THREADS=128


aprun -n $((${COBALT_PARTSIZE} * ${PPN})) -N ${PPN} -cc depth -d ${NUM_THREADS} -j 2 -e OMP_NUM_THREADS=${NUM_THREADS} \
python /projects/datascience/hsharma/bnn_hvd/src/GW_BNN/run_distributed_hvd.py \
 --fixed_var=0.0001 \
 --num_mc=500 \
 --alpha_KL=0.1 \
 --hvd=True \
 --lr=1e-4 \
 --batchsize=16 \
 --pout=True \
 --Datasave_path='/projects/datascience/hsharma/bnn_hvd/src/GW_BNN/results/data_store' \
 --TestIter=2000 \
 --PrintStep=10 \
 --num_iter=8000 \
 --Model_Name='spin_BNN_Model_Test_1' \
