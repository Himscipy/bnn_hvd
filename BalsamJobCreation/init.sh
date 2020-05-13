#!/bin/bash

#This script sets the balsam applications for running BNN .

# It is all based on running deep learning frameworks with python, so expects to use the intel python
# that comes with 'module load datascience/ ...'

# Every profiling application needs a name and executable for the core operation:
export APPLICATIONNAME="BNN_ThetaRun"
export APPLICATIONEXEC="python"
export BASEDIR=$(realpath $(dirname $0))


balsam app --name "${APPLICATIONNAME}" --exec ${APPLICATIONEXEC}


this_id=$(balsam ls apps | grep "${APPLICATIONNAME}" | awk '{print $1}')
balsam modify --type apps ${this_id} envscript ${BASEDIR}/setup.sh

