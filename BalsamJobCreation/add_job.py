from balsam.launcher import dag
import numpy as np

# Generic script for adding jobs to the balsam database.
APPLICATIONNAME="BNN_ThetaRun"

PathPythonCode = '/projects/datascience/hsharma/bnn_hvd/src/CNN_BNN_Model.py'
Data_Dir = '/projects/datascience/hsharma/bnn_hvd/DATA/mnist.npz'


generic_params =('{} --UseEpoch=False --epochs=10'
                 '--data_dir={} --model_data=./results/{}/ --model_type=BNN '
                 '--print_step=100 --iteration=1000 --learning_rate=0.001 --verbose=False'
                 '--batch_size=128 --bnnConv=BNN_conv_flip --cnnConv=CNN_conv '
                 '--num_intra=128 --num_inter=1')


Num_nodes = [1,2,4,8]

application = APPLICATIONNAME
workflow = APPLICATIONNAME + "_Runsof_Total_{}_jobs".format(str(len(Num_nodes)))

for i,node in enumerate (Num_nodes):
    model_name = 'BNN_Nodes_{}_Run_ID_{}'.format(node,i)

    args = generic_params.format(PathPythonCode,Data_Dir,model_name)

    print (args)
    job = dag.add_job(
                 name                = f'{application}_node{node}_BNNRun_ID_{i}',
                 workflow            = workflow,
                 description         = f'Run for different {node}',
                 num_nodes           = node,
                 ranks_per_node      = 1,
                 threads_per_rank    = 128,
                 threads_per_core    = 2,
                 cpu_affinity        = 'depth',
                 args                = args,
                 application         = application)
    
    job.data['node'] = node
    job.data['ID'] = i
    job.save()


# Here, generate a suggested balsam submission command
print("Example of a balsam submission command to run all of these jobs: ")
print("balsam submit-launch -n <num_nodes> -q <queue> -t <time> -A <account> --wf-filter {} --job-mode mpi".format(APPLICATIONNAME))
