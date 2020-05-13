#!/bin/bash

echo "Running the core setup script for this job"

# Typically you want to do this so you don't interfere between balsam and your jobs
unset PYTHONPATH

# This is how I set up my applications, you should modify this as needed.

# Load 

module load /soft/datascience/tensorflow/tensorflow-1.14-py36-hpctw



export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:$PYTHONPATH


module list

env > env_running.logs
echo "Done running the core setup script for this job"

