#!/bin/bash
#COBALT -t 1:00:00
#COBALT -n 2
#COBALT -A datascience
#COBALT -q debug-flat-quad 


# module load datascience/tensorflow-1.14
# module load datascience/horovod-0.18.1
# module load datascience/h5py-2.9.0
# module load /projects/datascience/hsharma/bnn_hvd/src/GW_BNN/run_scripts/tensorflow-1.14-HVD

module load /soft/datascience/tensorflow/tensorflow-1.14-py36-hpctw


export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:$PYTHONPATH

which python

PPN=1
NUM_THREADS=128


aprun -n $((${COBALT_PARTSIZE} * ${PPN})) -N ${PPN} -cc depth -d ${NUM_THREADS} -j 2 -e OMP_NUM_THREADS=${NUM_THREADS} \
python /projects/datascience/hsharma/bnn_hvd/src/CNN_BNN_Model.py --flagfile=config.cfg \
