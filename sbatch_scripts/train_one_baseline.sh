#!/bin/bash

# ##############################################################################
# NOTE: This script needs to be executed in the root dir of the repository!
# ##############################################################################

# TO CHANGE:
# - job name
# - output and error file paths
# - python command at the end

#SBATCH --job-name=baseline
#SBATCH --output=/vol/miltank/projects/practical_WS2425/cascade_learning/slurm_out/baseline-%A.out  # Standard output of the script (%A adds the job id)
#SBATCH --error=/vol/miltank/projects/practical_WS2425/cascade_learning/slurm_out/baseline-%A.err  # Standard error of the script (%A adds the job id)
#SBATCH --time=2-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=24  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=48G  # Memory in GB (Don't use more than 48GB per GPU unless you absolutely need it and know what you are doing)
#SBATCH --qos=master-queuesave

echo "Running experiment: config_one_stage_baseline"

export PYTHONUNBUFFERED=true

# Run the training script
python train.py --config_path train_configs/config_one_stage_baseline.json
