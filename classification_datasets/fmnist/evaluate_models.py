from utils.train import evaluate_ensemble, evaluate_saved_models, get_datasets
import sys
import getopt
from hydra import initialize, compose

#path_to_model = './multirun/2023-02-14/11-31-14/0/'


def get_conf_and_evaluate(argv):
    """
    How to use
    python evaluate_models --path 'multirun/2023-02-20/train_standard_lenet/' --mode 'ensemble'
    python evaluate_models --path 'multirun/2023-02-20/train_standard_mlp/' --mode 'ensemble'
    python evaluate_models --path 'multirun/2023-02-21/train_diverse_lenet/' --mode 'ensemble'
    python evaluate_models --path 'multirun/2023-02-21/train_diverse_mlp/' --mode 'ensemble'
    python evaluate_models --path 'multirun/2023-02-14/11-31-14/0/' --mode 'single'
    Parameters
    ----------
    argv

    Returns
    -------

    """

    opts, args = getopt.getopt(argv, "i:o:", ["path=", "mode="])

    for opt, arg in opts:
        if opt in ('-i', '--path'):
            path_to_model = arg
        elif opt in ('-o', '--mode'):
            mode = arg

    if mode == 'single':
        path_config = path_to_model+".hydra/"
    elif mode == 'ensemble':
        path_config = path_to_model + "/0/.hydra/"

    with initialize(version_base=None, config_path=path_config):
        # config is relative to a module
        cfg = compose(config_name="config")

    train_ds, test_ds, unlabeled_ds, validation_ds = get_datasets(cfg=cfg)
    if mode == 'single':
        res = evaluate_saved_models(path_to_model + '/ckpts/checkpoint_0', cfg=cfg, split_ds=test_ds)
    elif mode == 'ensemble':
        res = evaluate_ensemble(path_to_model, cfg=cfg, split_ds=test_ds)
    print(res)
    return res


if __name__ == '__main__':
    get_conf_and_evaluate(sys.argv[1:])




