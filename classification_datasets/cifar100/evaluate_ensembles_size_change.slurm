#!/bin/bash
#SBATCH --job-name=evaluate_ensemble_size_cifar100          # nom du job
##SBATCH --partition=gpu_p4          # decommenter pour la partition gpu_p4 (GPU A100 40 Go)
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --time=2:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=evaluate_ensemble_size_cifar100%j.out      # nom du fichier de sortie
#SBATCH --error=evaluate_ensemble_size_cifar100%j.out       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=1-10%10

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load python/3.9.12
conda init
source ~/.bash_profile
conda activate flax

# Execution du code
echo "The ensemble size is"
echo ${SLURM_ARRAY_TASK_ID}

echo "The architecture is lenet"
python evaluate_models.py --path 'multirun/train_standard_lenet/0/' --mode 'ensemble' --ensemble_size ${SLURM_ARRAY_TASK_ID}
python evaluate_models.py --path 'multirun/train_diverse_lenet/0/' --mode 'ensemble' --ensemble_size ${SLURM_ARRAY_TASK_ID}

echo "The architecture is mlp"
python evaluate_models.py --path 'multirun/train_standard_mlp/0/' --mode 'ensemble' --ensemble_size ${SLURM_ARRAY_TASK_ID}
python evaluate_models.py --path 'multirun/train_diverse_mlp/0/' --mode 'ensemble' --ensemble_size ${SLURM_ARRAY_TASK_ID}

echo "The architecture is resnet22"
python evaluate_models.py --path 'multirun/train_standard_resnet22/0/' --mode 'ensemble' --ensemble_size ${SLURM_ARRAY_TASK_ID}
python evaluate_models.py --path 'multirun/train_diverse_resnet22/0/' --mode 'ensemble' --ensemble_size ${SLURM_ARRAY_TASK_ID}
