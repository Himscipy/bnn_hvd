#!/bin/bash
#COBALT -t 1:00:00
#COBALT -n 4
#COBALT -A datascience
#COBALT -q debug-flat-quad 


module load /soft/environment/modules/modulefiles/datascience/tensorflow-1.13
export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:$PYTHONPATH

#HOROVOD_TIMELINE=/projects/datascience/hsharma/GW_Project/Single_Script/run_scripts/1_hvd/timeline.json

PPN=1
export NUM_THREADS=128


aprun -n $((${COBALT_PARTSIZE} * ${PPN})) -N ${PPN} -cc depth -d ${NUM_THREADS} -j 2 -e OMP_NUM_THREADS=${NUM_THREADS} \
/opt/intel/python/2017.0.035/intelpython35/bin/python \
 /projects/datascience/hsharma/GW_Project/Single_Script/run_distributed_hvd.py \
 --fixed_var=0.0001 \
 --num_mc=500 \
 --alpha_KL=0.1 \
 --hvd=True \
 --lr=1e-4 \
 --batchsize=16 \
 --pout=True \
 --Datasave_path='/projects/datascience/hsharma/bnn_hvd/GW_BNN/results/data_store' \
 --TestIter=2000 \
 --PrintStep=10 \
 --num_iter=8000 \
 --Model_Name='spin_BNN_Model_Test_1' \