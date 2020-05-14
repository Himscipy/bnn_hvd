#!/bin/bash
#COBALT -t 1:00:00
#COBALT -n 1
#COBALT -A mmaADSP
#COBALT -q debug-flat-quad 


module load /soft/environment/modules/modulefiles/datascience/tensorflow-1.13
export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:$PYTHONPATH


PPN=1
export NUM_THREADS=128

aprun -n $((${COBALT_PARTSIZE} * ${PPN})) -N ${PPN} -cc depth -d ${NUM_THREADS} -j 2 -e OMP_NUM_THREADS=${NUM_THREADS} \
python /projects/datascience/hsharma/bnn_hvd/src/GW_BNN/run_spin_omega_train.py \
 --fixed_var=0.0001 \
 --num_mc=100 \
 --alpha_KL=0.1 \
 --hvd=True \
 --pout=True \
 --Datasave_path='/projects/datascience/hsharma/data_store' \
 --TestIter=200 \
 --PrintStep=200 \
