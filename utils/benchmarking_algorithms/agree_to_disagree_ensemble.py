from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import os
from utils.train import create_train_state, cross_entropy_loss, compute_metrics_jitable, eval_model, evaluate_saved_models
from utils.datasets import dataset_num_classes, get_simple_ood_datasets
from flax import linen as nn
from jax import jit
from functools import partial


def train_agree_to_disagree_ensemble(cfg: DictConfig):
    """

    Parameters
    ----------
    cfg : DictConfig
        The configuration file for the experiment.
    Returns
    -------
    test_accuracy : float
        The final test accuracy of the trained model. This is useful when doing hyperparameter search with optuna.

    """
    if cfg.hyperparameters.dataset_name == 'fashion_mnist_dominoes' or \
            cfg.hyperparameters.dataset_name == 'Cifar10_dominoes':
        train_ds, test_ds, unlabeled_ds, validation_ds = get_simple_ood_datasets(cfg)
    elif False:
        a=1

    rng = jax.random.PRNGKey(0)#0
    rng, init_rng = jax.random.split(rng)
    directory = os.getcwd()
    states = []
    for i in range(cfg.hyperparameters.ensemble_size):
        states.append(create_train_state(init_rng, cfg))
        rng_dummy, init_rng = jax.random.split(init_rng)
    for member in range(cfg.hyperparameters.ensemble_size):
        new_state, init_rng = train_ensemble_member(states[member],
                                                    states[0:member],
                                                    train_ds,
                                                    unlabeled_ds,
                                                    validation_ds,
                                                    init_rng,
                                                    cfg)
        states[member] = new_state

    '''
    if cfg.hyperparameters.summary:
        validation_log_dir = '/logs/validation'
        validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)
    '''

    for i, state in zip(range(cfg.hyperparameters.ensemble_size), states):
        if not os.path.exists(str(i) + '/' + cfg.hyperparameters.CKPT_DIR):
            os.mkdir(str(i))
            os.mkdir(str(i) + '/' + cfg.hyperparameters.CKPT_DIR)
        else:
            if os.path.exists(str(i) + '/' + cfg.hyperparameters.CKPT_DIR + '/checkpoint_0'):
                os.remove(str(i) + '/' + cfg.hyperparameters.CKPT_DIR + '/checkpoint_0')
        checkpoints.save_checkpoint(ckpt_dir=str(i) + '/' + cfg.hyperparameters.CKPT_DIR, target=state, step=0)

    validation_metrics = evaluate_ensemble(directory, validation_ds, cfg)
    validation_metrics = jax.device_get(validation_metrics)
    print('final validation loss: %.2f, accuracy: %.2f, ECE: %.2f, TACE: %.2f, Brier: %.2f' % (
        validation_metrics['loss'],
        validation_metrics['accuracy']*100,
        validation_metrics['ece'],
        validation_metrics['tace'],
        validation_metrics['brier']))

    if math.isnan((1 - validation_metrics['accuracy']) ** 2 + validation_metrics['tace'] ** 2):
        return 10000
    else:
        return (1 - validation_metrics['accuracy']) ** 2 + validation_metrics['tace'] ** 2


def train_ensemble_member(current_member, previous_members, train_ds, unlabeled_ds, validation_ds, rng, cfg):

    for epoch in range(1, cfg.hyperparameters.epochs+1):
        # Use a separate PRNG key to permute image data during shuffling
        current_member, rng = train_epoch_agree_to_disagree(current_member, previous_members, train_ds, unlabeled_ds, rng, cfg)

        # Evaluate on the validation set after each training epoch
        logits_total, labels_total = eval_model(current_member, validation_ds)
        validation_metrics = compute_metrics_jitable(logits=logits_total, labels=labels_total,
                                                     num_classes=dataset_num_classes[cfg.hyperparameters.dataset_name])
        validation_metrics = jax.device_get(validation_metrics)

        print('validation epoch batched: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, validation_metrics['loss'],
            validation_metrics['accuracy']*100))

    return current_member, rng


def train_epoch_agree_to_disagree(current_member, previous_members, train_ds, unlabeled_ds, rng, cfg):
    """
    Train for a single epoch.
    Parameters
    ----------
    states : train_state.TrainState
        The training state of the experiment.
    train_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the training set.
    rng : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the random initialization of the neural network.
    cfg : DictConfig
        The configuration file for the experiment.

    Returns
    -------
    state : train_state.TrainState
        The new training state of the experiment.

    """
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // cfg.hyperparameters.batch_size_train

    unlabeled_ds_size = len(unlabeled_ds['image'])
    batch_size_full = unlabeled_ds_size // steps_per_epoch

    perms_train = jax.random.permutation(rng, train_ds_size)
    perms_train = perms_train[:steps_per_epoch*cfg.hyperparameters.batch_size_train]
    perms_train = perms_train.reshape((steps_per_epoch, cfg.hyperparameters.batch_size_train))

    if cfg.hyperparameters.unlabeled_batch_size == 'same':
        batch_size_unlabeled = cfg.hyperparameters.batch_size_train
    elif cfg.hyperparameters.unlabeled_batch_size == 'full':
        batch_size_unlabeled = batch_size_full

    rng, sub_rng = jax.random.split(rng)
    perms_unlabeled = jax.random.permutation(sub_rng, unlabeled_ds_size)
    if steps_per_epoch*batch_size_unlabeled <= perms_unlabeled.shape[0]:
        perms_unlabeled = perms_unlabeled[:steps_per_epoch*batch_size_unlabeled]
    else:
        fit = int((steps_per_epoch*batch_size_unlabeled) / perms_unlabeled.shape[0])
        rem = ((steps_per_epoch*batch_size_unlabeled) % perms_unlabeled.shape[0])
        perms_unlabeled = jnp.concatenate(fit*[perms_unlabeled])
        perms_unlabeled = jnp.concatenate([perms_unlabeled, perms_unlabeled[0:rem]])
    perms_unlabeled = perms_unlabeled.reshape((steps_per_epoch, batch_size_unlabeled ))

    for perm_both in zip(perms_train, perms_unlabeled):
        batch_train = {k: v[perm_both[0], ...] for k, v in train_ds.items()}
        batch_unlabeled = {k: v[perm_both[1], ...] for k, v in unlabeled_ds.items()}
        rng, sub_rng = jax.random.split(rng)
        current_member = train_step_agree_to_disagree(current_member,
                                                      previous_members,
                                                      batch_train,
                                                      batch_unlabeled,
                                                      sub_rng,
                                                      dataset_num_classes[cfg.hyperparameters.dataset_name],
                                                      cfg.hyperparameters.alpha)

    return current_member, rng


@partial(jit, static_argnames=['num_classes'])
def loss_a2d(params, state, dropout_rng, batch_x, batch_y, num_classes):
    def vmapped(x, y):
        logits = state.apply_fn(params, x, train=True, rngs={'dropout': dropout_rng})
        return cross_entropy_loss(logits=logits, labels=y, num_classes=num_classes)

    losses = jax.vmap(vmapped)(batch_x, batch_y)
    return jnp.mean(losses, axis=0)


@jax.jit
def probs_unlabeled_notrain(params, state, dropout_rng, batch_x):
    def vmapped(x):
        logits = state.apply_fn(params, x, train=False, rngs={'dropout': dropout_rng})
        return logits
    logits = jax.vmap(vmapped)(batch_x)
    return nn.softmax(logits, axis=1)


@jax.jit
def probs_unlabeled_train(params, state, dropout_rng, batch_x):
    def vmapped(x):
        logits = state.apply_fn(params, x, train=True, rngs={'dropout': dropout_rng})
        return logits
    logits = jax.vmap(vmapped)(batch_x)
    return nn.softmax(logits, axis=1)


def train_step_agree_to_disagree(current_member,
                                 previous_members,
                                 batch_train,
                                 batch_unlabeled,
                                 rng,
                                 num_classes,
                                 alpha):
    """
    Trains the neural network for a single step."
    Parameters
    ----------
    states : train_state.TrainState
        The initial training state of the experiment.
    batch : dict
        Dictionary with keys 'image' and 'label' corresponding to a batch of the training set.
    dropout_rngs : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the randomness of the dropout layers.
    num_classes: int
        The number of classes in the classification problem.

    Returns
    -------
    state : train_state.TrainState
        The training state of the experiment.
    metrics : dict
        A python dictionary with keys "loss" and "accuracy" corresponding to the cross-entropy loss and the accuracy
        for some logits and labels.
    new_dropout_rng : jax.random.PRNGKey
        New pseudo-random number generator (PRNG) key for the randomness of the dropout layers.

    """

    grad_fn = jax.grad(loss_a2d, has_aux=False)

    rng, dropout_rng = jax.random.split(rng)
    grads = grad_fn(current_member.params,
                           current_member,
                           dropout_rng,
                           batch_train['image'],
                           batch_train['label'],
                           num_classes)

    probs_previous = []
    for state in previous_members:
        rng, dropout_rng = jax.random.split(rng)
        probs_previous.append(probs_unlabeled_notrain(state.params,
                                    state,
                                    dropout_rng,
                                    batch_unlabeled['image']))

    def diversity_loss(params, state, dropout_rng, batch_x):

        probs_current = probs_unlabeled_train(params,
                                        state,
                                        dropout_rng,
                                        batch_x)
        highest = jnp.argmax(probs_current, axis=1)
        ind_tmp = jnp.arange(probs_current.shape[0])
        highest = jnp.array([ind_tmp, highest])
        linear_highest = jnp.ravel_multi_index(highest, probs_current.shape)
        index_top = jnp.zeros(shape=probs_current.shape)
        index_top = index_top.reshape((-1)).at[linear_highest].set(1).reshape(index_top.shape)
        index_bottom = jnp.ones(shape=probs_current.shape)
        index_bottom = index_bottom.reshape((-1)).at[linear_highest].set(0).reshape(index_bottom.shape)
        top_current = jnp.sum(probs_current, axis=1, where=index_top)
        bottom_current = jnp.sum(probs_current, axis=1, where=index_bottom)

        div_loss = 0.0
        for pr in probs_previous:
            top_previous = jnp.sum(pr, axis=1, where=index_top)
            bottom_previous = jnp.sum(pr, axis=1, where=index_bottom)

            div_loss -= (1/len(probs_previous))*jnp.sum(jnp.log(top_current*bottom_previous+top_previous*bottom_current))

        return div_loss

    grad_fn_diversity = jax.grad(diversity_loss, has_aux=False)
    rng, dropout_rng = jax.random.split(rng)
    grads_diverse = grad_fn_diversity(current_member.params,
                                     current_member,
                                     dropout_rng,
                                     batch_unlabeled['image'])

    grads_total = jax.tree_map(lambda x, y: x + alpha*y, grads, grads_diverse)
    current_member = current_member.apply_gradients(grads=grads_total)
    return current_member


def evaluate_ensemble(path_to_ensemble, split_ds, cfg):
    """
    Gets a path to an ensemble and evaluates its metrics.
    Parameters
    ----------
    path_to_ensemble: str
        The path to different ensemble members trained through hydra.
    split_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to some evaluation set.
    cfg : DictConfig
        The configuration file for the experiment.

    Returns
    -------
    validation_metrics: Dict
        A dictionary with elements

    """
    subdirectories = os.listdir(path_to_ensemble)
    paths = []
    for dir in subdirectories:
        if dir.isnumeric():
            if not cfg.hyperparameters.neglect_first_ensemble_member or not dir == '0':
                paths.append(path_to_ensemble + '/' + dir + '/ckpts/checkpoint_0')




    validation_metrics = evaluate_saved_models(paths, split_ds=split_ds, cfg=cfg)

    return validation_metrics