#!/bin/bash
#SBATCH --job-name=fmnist_tune_diverse_resnet22          # nom du job
#SBATCH --partition=gpu_p4          # decommenter pour la partition gpu_p4 (GPU A100 40 Go)
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=fmnist_tune_diverse_resnet22%j.out      # nom du fichier de sortie
#SBATCH --error=fmnist_tune_diverse_resnet22%j.out       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=1-20%10

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load python/3.9.12
conda init
source ~/.bash_profile
conda activate flax

# Execution du code
python train_models.py --multirun hyperparameters=diverse server=jeanzay hyperparameters.delete_checkpoints=True hyperparameters.model='FixupWideResNet22_SmallInputs' hyperparameters.epochs=10,20,50,80,100,200,300 optimizer.learning_rate=0.0001,0.001 hyperparameters.mylambda=1,10,100,1000,10000 hyperparameters.prior_var=0.1,0.01,0.001,0.0001,0.00001,0.000001 hyperparameters.beta=0.1,0.01,0.001,0.0001,0.00001,0.000001 hydra.sweeper.direction=minimize hydra.job.chdir=True hydra/sweeper=optuna hydra.sweeper.n_trials=1 hydra.sweep.dir='multirun/tune_diverse_resnet22_'${SLURM_ARRAY_TASK_ID}