# Balsam Job Creation Scripts

Balsam is a work schecduler on theta. The details about using it on theta can be found in the [documentation](https://balsam.readthedocs.io/en/latest/). The python scripts present here will help in creating and submitting BNN model jobs on Theta. 


- Step 0 : These are intial balsam steps (skip if you already have a Balsam Env. setup)
  + `module load balsam`
  + `balsam init ./Tut_BalsamBNN `
  + `source balsamactivate /lus/theta-fs0/projects/datascience/hsharma/bnn_hvd/BalsamJobCreation/Tut_BalsamBNN`
  + `balsam submit-launch -n <num_nodes> -q <queue> -t <time> -A <account> --wf-filter BNN_ThetaRun --job-mode mpi` 

- **Step 1:**
    + Run `./init.sh`

- **Step 2:**
    + Modifiy line-7 and line-8 in the `add_job.py` script these lines provide path to the source code and the data path. You can change other parameters as well in the string defination of the variable `generic_params` according to the requriments. Once edited run;     

        ```
          python add_job.py
        ```  
      
      This will create Balsam jobs in your balsam database. The program will return `balsam launch` instructions which can be used to submit the jobs on Theta.

- **Step 3:** 

    The submission command for running the jobs created in the workflow.

    ```
    balsam submit-launch -n <num_nodes> -q <queue> -t <time> -A <account> --wf-filter BNN_ThetaRun --job-mode mpi
    ``` 
    
## Tips for Balsam;
 + Deleting the jobs based on workflow from command line.
    ```
    python -c 'from balsam.launcher.dag import BalsamJob; BalsamJob.objects.filter(workflow="myWorkflow").delete()'
    ``` 



**Note**: 
This submission and creation procedure was motivated by Corey Adams use of Balsam for performing profiling with vtune. 







