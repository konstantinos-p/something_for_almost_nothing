from utils.train import evaluate_ensemble, evaluate_saved_models, get_cifar10_corrupted, cifar10_corruptions, cifar10_corruption_severities
import sys, os
import getopt
from hydra import initialize, compose
import numpy as np
import pickle

#path_to_model = './multirun/2023-02-14/11-31-14/0/'


def get_conf_and_evaluate(argv):
    """
    How to use

    *** This fails silently when the correct folder is not given. ***

    python evaluate_models.py --path 'multirun/2023-02-20/train_standard_lenet/' --mode 'ensemble'
    python evaluate_models.py --path 'multirun/2023-02-20/train_standard_mlp/' --mode 'ensemble'
    python evaluate_models.py --path 'multirun/2023-02-21/train_diverse_lenet/' --mode 'ensemble'
    python evaluate_models.py --path 'multirun/2023-02-21/train_diverse_mlp/' --mode 'ensemble'
    python evaluate_models.py --path 'multirun/2023-02-14/11-31-14/0/' --mode 'single'

    Parameters
    ----------
    argv

    Returns
    -------

    """

    if not os.path.exists('test_results_on_corrupted_data'):
        os.mkdir('test_results_on_corrupted_data')
    else:
        raise FileExistsError('The results folder already exists.')

    opts, args = getopt.getopt(argv, "i:o:f:c:s:", ["path=", "mode=", "ensemble_size=", "corruption=", "max_severity="])

    ensemble_size = 0
    corruptions = cifar10_corruptions
    severities = cifar10_corruption_severities
    metric = 'tace'

    for opt, arg in opts:
        if opt in ('-i', '--path'):
            path_to_model = arg
        elif opt in ('-o', '--mode'):
            mode = arg
        elif opt in ('-f', '--ensemble_size'):
            ensemble_size = int(arg)
        elif opt in ('-c', '--corruption'):
            corruptions = [arg]
        elif opt in ('-s', '--max_severity'):
            severities = int(arg)

    if mode == 'single':
        path_config = path_to_model+".hydra/"
    elif mode == 'ensemble':
        path_config = path_to_model + ".hydra/"#/0/.hydra/

    with initialize(version_base=None, config_path=path_config):
        # config is relative to a module
        cfg = compose(config_name="config")

    results = np.zeros((len(severities), len(corruptions)))

    for severity, i in zip(severities, range(len(severities))):
        for corruption, j in zip(corruptions, range(len(corruptions))):

            test_ds = get_cifar10_corrupted(corruption=corruption,
                                            severity=severity,
                                            dataset_dir=cfg.server.dataset_dir
                                            )

            if mode == 'single':
                res = evaluate_saved_models(path_to_model + '/ckpts/checkpoint_0', cfg=cfg, split_ds=test_ds)
            elif mode == 'ensemble':
                if ensemble_size == 0:
                    res = evaluate_ensemble(path_to_model,
                                            cfg=cfg,
                                            split_ds=test_ds)
                else:
                    res = evaluate_ensemble(path_to_model,
                                            cfg=cfg,
                                            split_ds=test_ds,
                                            ensemble_size=ensemble_size)

            results[i, j] = res[metric]


    np.save('test_results_on_corrupted_data/results.npy', results)
    with open("test_results_on_corrupted_data/corruptions", "wb") as fp:  # Pickling
        pickle.dump(corruptions, fp)
    with open("test_results_on_corrupted_data/severities", "wb") as fp:  # Pickling
        pickle.dump(severities, fp)

    mean_results = np.mean(results, axis=1)
    print('The results of the evaluation, with increasing severity are:')
    print(mean_results)

    print('Corruptions tested: '+' '.join(' '+corruption for corruption in corruptions))
    print('Severity levels tested: '+' '.join(' '+str(sev) for sev in severities))

    return


if __name__ == '__main__':
    get_conf_and_evaluate(sys.argv[1:])




