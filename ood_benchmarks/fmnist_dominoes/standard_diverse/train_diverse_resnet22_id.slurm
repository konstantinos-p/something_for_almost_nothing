#!/bin/bash
#SBATCH --job-name=fmnist_train_diverse_resnet22          # nom du job
#SBATCH --partition=gpu_p4          # decommenter pour la partition gpu_p4 (GPU A100 40 Go)
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=fmnist_train_diverse_resnet22%j.out      # nom du fichier de sortie
#SBATCH --error=fmnist_train_diverse_resnet22%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load python/3.9.12
conda init
source ~/.bash_profile
conda activate flax

# Execution du code
python -u train_models.py --multirun hyperparameters.size_training=1000 hyperparameters=diverse server=jeanzay hyperparameters.model='FixupWideResNet22' hyperparameters.ensemble_size=10 optimizer=adamw hyperparameters.epochs=80 optimizer.learning_rate=0.001 hyperparameters.mylambda=10 hyperparameters.prior_var=0.00001 hyperparameters.beta=0.01 hydra.sweeper.direction=minimize hydra.job.chdir=True hydra/sweeper=optuna hydra/sweeper/sampler=grid hydra.sweep.dir='multirun/train_diverse_resnet22'