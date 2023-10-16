<h2 align="center">Something for (almost) nothing.</h2>

We present a method to improve the calibration of deep ensembles in the small training data regime in the presence of unlabeled data. Our approach is extremely simple to implement: given an unlabeled set, for each unlabeled data point, we simply fit a different randomly selected label with each ensemble member. We provide a theoretical analysis based on a PAC-Bayes bound which guarantees that if we fit such a labeling on unlabeled data, and the true labels on the training data, we obtain low negative log-likelihood and high ensemble diversity on testing samples. Empirically, through detailed experiments, we find that for low to moderately-sized training sets, our ensembles are more diverse and provide better calibration than standard ensembles, sometimes significantly.

$\nu$-ensembles are trained based on the following algorithm

<p align="center">
    <img src="/assets/img/nu_ensembles.png"/>
</p>

Below we see some results on the CIFAR-10 and CIFAR-100 datasets

<p align="center">
    <img src="/assets/img/cifar10_cifar100.png"/>
</p>

<h2> :bookmark: Usage.</h2>

- `utils/models` implements all the models used for the evaluation
- `utils/metrics` implements all the metrics used for the evaluation
- `utils/train` implements both the $\nu$-ensembles and standard ensembles
- `utils/benchmarking_algorithms` impements dice-ensembles, masegosa ensembles and agree to disagree ensembles 

The code is currently launched from `ablation_studies` and `ablation_studies_other_benchmarks`.

The experiment folders are structured as `training_set_#1@#2_..._ens_size#3` where `#1` is the size of the training set
`#2` is the size of the unlabeled set, and `#3` is the size of the ensemble. Experimental runs are configured through 
**hydra**. The configuration files exist in the `conf` folders of each experiment.


- `train_models.py` trains a model according to the **hydra** configuration in `conf`. 
- `evaluate_models.py` evaluates trained models on the test set
- `evaluate_corruptions.py` evaluates trained models on common image corruptions with 5 levels of intensity (this script works only for the CIFAR-10
dataset).

The hyperparameter ranges used to recreate the experiments of the paper can be found as `train_....slurm` and `tune_....slurm`
files in each experiment folder. `tune_....slurm` files were used for optimizing the hyperparameters (without storing the weights), while `train_....slurm`
were used to store the final weights.

An example python command to train a LeNet architecture with the $\nu$-ensembles algorithm is

```
python train_models.py --multirun hyperparameters=diverse server=jeanzay hyperparameters.model='LeNet' optimizer=adamw hyperparameters.epochs=100 optimizer.learning_rate=0.001 hyperparameters.beta=0.01 hydra.sweeper.direction=minimize hydra.job.chdir=True hydra/sweeper=optuna hydra/sweeper/sampler=grid hydra.sweep.dir='multirun/train_diverse_lenet'
```

note that in the above command, even though the **Optuna** hyperparameter is invoked only a single instance is given for each hyperparameter. Thus only
a single network is finally trained.

<h2> :memo: Citation </h2>

When citing this repository on your scientific publications please use the following **BibTeX** citation:

```bibtex
@article{pitas2023something,
  title={Something for (almost) nothing: Improving deep ensemble calibration using unlabeled data},
  author={Pitas, Konstantinos and Arbel, Julyan},
  journal={arXiv preprint arXiv:2310.02885},
  year={2023}
}

```

<h2> :envelope: Contact Information </h2>
You can contact me at any of my social network profiles:

- :briefcase: Linkedin: https://www.linkedin.com/in/konstantinos-pitas-lts2-epfl/
- :octocat: Github: https://github.com/konstantinos-p

Or via email at cwstas2007@hotmail.com

<h2> :warning: Disclaimer </h2>
This Python package has been made for research purposes.

