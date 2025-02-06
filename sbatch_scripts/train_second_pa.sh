#!/bin/bash

# ##############################################################################
# NOTE: This script needs to be executed in the root dir of the repository!
# ##############################################################################

# TO CHANGE:
# - job name
# - output and error file paths
# - python command at the end

#SBATCH --job-name=second_pa_pred_sup-dev
#SBATCH --output=/vol/miltank/projects/practical_WS2425/cascade_learning/slurm_out/second_stage_pa_pred_sup-dev-%A.out  # Standard output of the script (%A adds the job id)
#SBATCH --error=/vol/miltank/projects/practical_WS2425/cascade_learning/slurm_out/second_stage_pa_pred_sup-dev-%A.err  # Standard error of the script (%A adds the job id)
#SBATCH --time=2-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=24  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=48G  # Memory in GB (Don't use more than 48GB per GPU unless you absolutely need it and know what you are doing)
#SBATCH --qos=master-queuesave

export PYTHONUNBUFFERED=true

# Run the training script
python train_new.py --config_path train_configs/config_two_stage_second_pa.json
