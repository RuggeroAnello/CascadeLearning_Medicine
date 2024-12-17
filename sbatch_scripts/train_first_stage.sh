#!/bin/bash


# ##############################################################################
# NOTE: This script needs to be executed in the root dir of the repository!
# ##############################################################################


# TO CHANGE:
# - job name
# - output and error file paths
# - python command at the end

#SBATCH --job-name=first_pred_ap-pa
#SBATCH --output=/vol/aimspace/projects/practical_WS2425/cascade_learning/slurm_out/first_pred_ap-pa-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=/vol/aimspace/projects/practical_WS2425/cascade_learning/slurm_out/first_pred_ap-pa-%A.err  # Standard error of the script
#SBATCH --time=2-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=24  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=48G  # Memory in GB (Don't use more than 48GB per GPU unless you absolutely need it and know what you are doing)

export PYTHONUNBUFFERED=true

conda init
echo "INITIALIZED CONDA"

# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
echo "DEACTIVATED CONDA ENVIRONMENT"

conda activate personalized_ml # If this does not work try 'source activate ptl'
echo "ACTIVATED 'personlized_ml' CONDA ENVIRONMENT"
# run the program
python train.py --model_type two_stage_first
                                           