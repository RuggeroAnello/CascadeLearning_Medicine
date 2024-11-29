#!/bin/bash
 
#SBATCH --job-name=chexp_non_p
#SBATCH --output=./out/chexp_non_p-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=./out/chexp_non_p-%A.err  # Standard error of the script
#SBATCH --time=2-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=24  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=80G  # Memory in GB (Don't use more than 48GB per GPU unless you absolutely need it and know what you are doing)

export PYTHONUNBUFFERED=true

# load python module
ml python/anaconda3
experimentName=$1
# # activate corresponding environment
# conda deactivate # If you launch your script from a terminal where your environment is already loaded conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
# conda activate indiv_privacy # If this does not work try 'source activate ptl'
echo "Running experiment: $experimentName"
# run the program
python idp_sgd/dpsgd_algos/individual_dp_sgd.py --config /vol/aimspace/users/kaiserj/Indivdiual_Privacy_DPSGD_Evaluation/yaml_files_training_settings/main_config_chexpert.yaml \
                                                --use_cuda \
                                                --use_wandb \
                                                --disable_inner_tqdm \
                                                --not_use_date \
                                                --experiment $1 \
                                                "${@:2}"
                                           