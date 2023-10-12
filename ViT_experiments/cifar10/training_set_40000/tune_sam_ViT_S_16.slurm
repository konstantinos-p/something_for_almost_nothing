#!/bin/bash
#SBATCH --job-name=cifar10_tune_sam_ViT_S_16          # nom du job
#SBATCH --partition=gpu_p4          # decommenter pour la partition gpu_p4 (GPU A100 40 Go)
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=cifar10_tune_sam_ViT_S_16%j.out      # nom du fichier de sortie
#SBATCH --error=cifar10_tune_sam_ViT_S_16%j.out       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=1-20%20

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load python/3.9.12
conda init
source ~/.bash_profile
conda activate flax

# Execution du code
python -u train_models.py --multirun hyperparameters=sam server=jeanzay hyperparameters.delete_checkpoints=True hyperparameters.model='ViT_S_16' optimizer=sgd optimizer.learning_rate=0.1,0.01,0.001,0.0001,0.00001 hyperparameters.rho_sam=0.01,0.02,0.05,0.07,0.1,0.11,0.12,0.15,0.17,0.2 hyperparameters.epochs=200,250,300,350,400,450,500 hydra.sweeper.direction=minimize hydra.job.chdir=True hydra/sweeper=optuna hydra.sweeper.n_trials=1 hydra.sweep.dir='multirun/tune_sam_ViT_S_16_'${SLURM_ARRAY_TASK_ID}