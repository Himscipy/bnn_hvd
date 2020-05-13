# Balsam Job Creation Scripts

Balsam is a work schecduler on theta. The details about using it on theta can be found in the [documentation](https://balsam.readthedocs.io/en/latest/). The python scripts present here will help in creating and submitting BNN model jobs on Theta. 

Note: Make sure you have run `module load balsam` and created the balsam database before running the steps below.

- Step 1: Run `./init.sh` first

- Step 2: Based on the source code path the `add_job.py` line-7, line-8 to provide correct path of the source code and the 
          data path. You can change other parameters as well in the `generic_params` according to the requriments. Once done
          run `python add_job.py`. This will create Balsam jobs in your balsam database. The program will return `balsam launch` instructions which can be used to submit the jobs on Theta.


## Tips for Balsam;
 + Deleting the jobs based on workflow from command line.
    - `python -c 'from balsam.launcher.dag import BalsamJob; BalsamJob.objects.filter(workflow="myWorkflow").delete()'` 




This submission and creation procedure was motivated by Corey Adams use of Balsam for performing profiling with vtune. 







