from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state, checkpoints
import numpy as np
import optax
import tensorflow_datasets as tfds
import tensorflow as tf
from typing import Any
from utils import models
import warnings
import os
from utils.metrics import get_ece, get_tace, get_brier
from flax.linen.activation import softmax
import chex

ModuleDef = Any
KeyArray = random.KeyArray
Array = Any
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any
RealNumeric = Any

dataset_dimensions = {
    'Cifar10': [32, 32, 3],
    'Cifar100': [32, 32, 3],
    'svhn_cropped': [32, 32, 3],
    'fashion_mnist': [28, 28, 1],
}

dataset_num_classes = {
    'Cifar10': 10,
    'Cifar100': 100,
    'svhn_cropped': 10,
    'fashion_mnist': 10,
}


def cross_entropy_loss(*, logits, labels):
    """
    The cross-entropy loss.
    Parameters
    ----------
    logits: float
        The prediction preactivations of the neural network.
    labels:
        The groundtruth labels.

    Returns
    -------
        : optax.softmax_cross_entropy
        The softmax_cross_entropy loss.

    """
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def cross_entropy_loss_no_softmax(*, logits, labels):
    """
    The cross-entropy loss.
    Parameters
    ----------
    logits: float
        The prediction activations of the neural network (after the softmax).
    labels:
        The groundtruth labels.

    Returns
    -------
        : optax.softmax_cross_entropy
        The softmax_cross_entropy loss.

    """
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return cross_entropy(logits=logits, labels=labels_onehot).mean()


def cross_entropy(logits: chex.Array, labels: chex.Array,) -> chex.Array:
    """
    Computes the cross entropy between sets of logits and labels.
    """
    chex.assert_type([logits], float)
    return -jnp.log(jnp.sum(labels * logits, axis=-1))


def compute_metrics(*, logits, labels):
    """
    Computes the crossentropy loss and the accuracy for a given set of predictions and groundtruth labels.
    Parameters
    ----------
    logits: float
        The predictions of the neural network (after the softmax).
    labels:
        The groundtruth labels.

    Returns
    -------
    metrics: dict
        A python dictionary with keys "loss", "accuracy", "ece", "tace" and "brier" for these different metrics which
        have been computed given some logits and labels.

    """
    loss = cross_entropy_loss_no_softmax(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    ece = get_ece(logits, labels)
    tace = get_tace(logits, labels)
    brier = get_brier(logits, labels)
    try:
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'ece': ece.item(),
            'tace': tace.item(),
            'brier': brier.item()
        }
    except:
        metrics = {
            'loss': 100,
            'accuracy': 0,
            'ece': 1,
            'tace': 1,
            'brier': 100
        }

    return metrics


def compute_metrics_jitable(*, logits, labels):
    """
    Computes the crossentropy loss and the accuracy for a given set of predictions and groundtruth labels.
    Parameters
    ----------
    logits: float
        The predictions of the neural network (after the softmax).
    labels:
        The groundtruth labels.

    Returns
    -------
    metrics: dict
        A python dictionary with keys "loss", "accuracy", "ece", "tace" and "brier" for these different metrics which
        have been computed given some logits and labels.

    """
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }

    return metrics


def get_datasets(cfg):
    """
    Load train and test datasets into memory.
    Parameters
    ----------
    cfg : DictConfig
        The configuration file for the experiment.

    Returns
    -------
    train_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the training set.
    test_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the test set.
    """
    if cfg.hyperparameters.dataset_name not in ['Cifar10', 'Cifar100', 'svhn_cropped', 'fashion_mnist']:
        warnings.warn(cfg.hyperparameters.dataset_name+' might not exist in tensorflow_datasets. These experiments have been created for datasets ``Cifar10``, ``Cifar100``, ``svhn_cropped`` and ``fashion_mnist``.')

    if cfg.server.dataset_dir == 'default':
        ds_builder = tfds.builder(cfg.hyperparameters.dataset_name)
        ds_builder.download_and_prepare()
        train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
        test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    else:
        train_ds = tfds.load(name=cfg.hyperparameters.dataset_name, data_dir=cfg.server.dataset_dir, split='train',
                             batch_size=-1)
        test_ds = tfds.load(name=cfg.hyperparameters.dataset_name, data_dir=cfg.server.dataset_dir, split='test',
                            batch_size=-1)

    train_ds['image'] = jnp.float32(train_ds['image'])/255.
    test_ds['image'] = jnp.float32(test_ds['image'])/255.

    train_ds['label'] = jnp.int32(train_ds['label'])
    test_ds['label'] = jnp.int32(test_ds['label'])

    train_ds = {i: train_ds[i] for i in train_ds if i != 'id'}
    test_ds = {i: test_ds[i] for i in test_ds if i != 'id'}

    validation_ds = {}
    unlabeled_ds = {}

    validation_ds['image'] = train_ds['image'][0:cfg.hyperparameters.size_validation]
    validation_ds['label'] = train_ds['label'][0:cfg.hyperparameters.size_validation]

    unlabeled_ds['image'] = train_ds['image'][cfg.hyperparameters.size_validation+cfg.hyperparameters.size_training:]
    unlabeled_ds['label'] = train_ds['label'][cfg.hyperparameters.size_validation+cfg.hyperparameters.size_training:]

    train_ds['image'] = train_ds['image'][cfg.hyperparameters.size_validation:cfg.hyperparameters.size_validation
                                                                              +cfg.hyperparameters.size_training]
    train_ds['label'] = train_ds['label'][cfg.hyperparameters.size_validation:cfg.hyperparameters.size_validation
                                                                              +cfg.hyperparameters.size_training]

    return train_ds, test_ds, unlabeled_ds, validation_ds


def create_train_state(rng, cfg):
    """
    Creates initial `TrainState`.
    Parameters
    ----------
    rng : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the random initialization of the neural network.
    cfg : DictConfig
        The configuration file for the experiment.

    Returns
    -------
        :train_state.TrainState
        The initial training state of the experiment.
    """
    model_cls = getattr(models, cfg.hyperparameters.model)
    network = model_cls(num_classes=dataset_num_classes[cfg.hyperparameters.dataset_name])
    params = network.init(rng, jnp.ones(dataset_dimensions[cfg.hyperparameters.dataset_name]), train=False)
    if cfg.optimizer.name == 'sgd':
        tx = optax.sgd(cfg.optimizer.learning_rate, cfg.optimizer.momentum)
    elif cfg.optimizer.name == 'adamw':
        tx = optax.adamw(cfg.optimizer.learning_rate)
    return train_state.TrainState.create(apply_fn=network.apply, params=params, tx=tx)


@jax.jit
def train_step_diverse(state, batch_training, batch_unlabeled, dropout_rng, mylambda, prior_var, beta):
    """
    Trains the neural network for a single step."
    Parameters
    ----------
    state : train_state.TrainState
        The initial training state of the experiment.
    batch_training : dict
        Dictionary with keys 'image' and 'label' corresponding to a batch of the training set.
    batch_unlabeled : dict
        Dictionary with keys 'image' and 'label' corresponding to a batch of the unlabeled set.
    dropout_rng : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the randomness of the dropout layers.

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
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    # Training set grads
    def per_sample_loss_and_logits(params, x, y):
        logits = state.apply_fn(params, x, train=True, rngs={'dropout': dropout_rng})
        return cross_entropy_loss(logits=logits, labels=y), logits
    train_loss_grad_fn = jax.vmap(jax.grad(per_sample_loss_and_logits, has_aux=True), (None, 0, 0))
    train_grads_per_sample, train_logits_per_sample = train_loss_grad_fn(state.params, batch_training['image'],
                                                                         batch_training['label'])

    # Unlabeled set grads
    def per_sample_logits(params, x):
        logits = state.apply_fn(params, x, train=True, rngs={'dropout': dropout_rng})
        return logits
    unlabeled_loss_jacrev_fn = jax.vmap(jax.jacrev(per_sample_logits), (None, 0))
    jacrev_per_sample = unlabeled_loss_jacrev_fn(state.params, batch_unlabeled['image'])

    def diversity_gradient(train_grads, unlabeled_jacobians, mylambda, prior_var):

        term1_pow2 = jax.tree_map(lambda x: jnp.mean((-x) ** 2, axis=0), train_grads)
        term1_pow3 = jax.tree_map(lambda x: jnp.mean((-x) ** 3, axis=0), train_grads)
        term1 = jax.tree_map(lambda x, y: -(mylambda * x + 1 / prior_var) ** (-2) * (2 * mylambda * y), term1_pow2,
                             term1_pow3)

        term2 = jax.tree_map(lambda x: jnp.mean(x ** 2, axis=[0, 1]), unlabeled_jacobians)

        term3 = jax.tree_map(lambda x: (mylambda * x + 1 / prior_var) ** (-1), term1_pow2)

        term4 = jax.tree_map(lambda x: jnp.mean(2 * x ** 3, axis=[0, 1]), unlabeled_jacobians)

        return jax.tree_map(lambda x, y, z, k: x * y + z * k, term1, term2, term3, term4)

    diversity_gradient_rescaled = jax.tree_map(lambda x: x * beta, diversity_gradient(train_grads_per_sample, jacrev_per_sample,
                                                                 mylambda, prior_var))
    '''
    Note that the diverse gradients that we compute should be subtracted from the normal ones.
    '''
    grads_total = jax.tree_map(lambda x, y: x - y, jax.tree_map(lambda x: jnp.sum(x, axis=0), train_grads_per_sample),
                               diversity_gradient_rescaled)

    state = state.apply_gradients(grads=grads_total)
    metrics = compute_metrics_jitable(logits=train_logits_per_sample, labels=batch_training['label'])
    return state, metrics, new_dropout_rng


@jax.jit
def train_step_standard(state, batch, dropout_rng):
    """
    Trains the neural network for a single step."
    Parameters
    ----------
    state : train_state.TrainState
        The initial training state of the experiment.
    batch : dict
        Dictionary with keys 'image' and 'label' corresponding to a batch of the training set.
    dropout_rng : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the randomness of the dropout layers.

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
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params, batch_x, batch_y):
        def per_sample_loss_and_logits(x, y):
            logits = state.apply_fn(params, x, train=True, rngs={'dropout': dropout_rng})
            return cross_entropy_loss(logits=logits, labels=y), logits
        total_loss, total_logits = jax.vmap(per_sample_loss_and_logits)(batch_x, batch_y)
        return jnp.mean(total_loss, axis=0), total_logits

    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, logits = grad_fn(state.params, batch['image'], batch['label'])
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics_jitable(logits=logits, labels=batch['label'])
    return state, metrics, new_dropout_rng


@jax.jit
def eval_step(state, batch):
    """
    A single evaluation step of the output logits of the neural network for a batch of inputs, as well as the
    cross-entropy loss and the classification accuracy.
    Parameters
    ----------
    state : train_state.TrainState
        The training state of the experiment.
    batch : dict
        Dictionary with keys 'image' and 'label' corresponding to a batch of the training set.

    Returns
    -------
    metrics : dict
        A python dictionary with keys "loss" and "accuracy" corresponding to the cross-entropy loss and the accuracy
        for some logits and labels.
    """

    def per_sample_logits(params, x):
        return state.apply_fn(params, x, train=False)
    return jax.vmap(per_sample_logits, (None, 0))(state.params, batch['image'])


def train_epoch_diverse(state, train_ds, unlabeled_ds, epoch, rng, cfg):
    """
    Train for a single epoch.
    Parameters
    ----------
    state : train_state.TrainState
        The training state of the experiment.
    train_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the training set.
    unlabeled_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the unlabeled set.
    epoch : int
        The number of the current epoch.
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

    perms_unlabeled = jax.random.permutation(rng, unlabeled_ds_size)
    perms_unlabeled = perms_unlabeled[:steps_per_epoch*batch_size_unlabeled]
    perms_unlabeled = perms_unlabeled.reshape((steps_per_epoch, batch_size_unlabeled ))

    batch_metrics = []
    dropout_rng = jax.random.split(rng, jax.local_device_count())[0]
    for perm_both in zip(perms_train, perms_unlabeled):
        batch_train = {k: v[perm_both[0], ...] for k, v in train_ds.items()}
        batch_unlabeled = {k: v[perm_both[1], ...] for k, v in unlabeled_ds.items()}
        state, metrics, dropout_rng = train_step_diverse(state, batch_train, batch_unlabeled, dropout_rng,
                                                         cfg.hyperparameters.mylambda, cfg.hyperparameters.prior_var,
                                                         cfg.hyperparameters.beta)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.train_state
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0] # jnp.mean does not work on lists
    }

    print('train epoch: %d, loss %.4f, accuracy: %.2f' % (epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy']*100))


    if cfg.hyperparameters.summary == True:
        train_log_dir = 'logs/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        with train_summary_writer.as_default():
            tf.summary.scalar('accuracy', epoch_metrics_np['accuracy'], step=epoch)

    return state


def train_epoch_standard(state, train_ds, epoch, rng, cfg):
    """
    Train for a single epoch.
    Parameters
    ----------
    state : train_state.TrainState
        The training state of the experiment.
    train_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the training set.
    batch_size : int
        The size of the batch.
    epoch : int
        The number of the current epoch.
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

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch*cfg.hyperparameters.batch_size_train]
    perms = perms.reshape((steps_per_epoch, cfg.hyperparameters.batch_size_train))
    batch_metrics = []
    dropout_rng = jax.random.split(rng, jax.local_device_count())[0]
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics, dropout_rng = train_step_standard(state, batch, dropout_rng)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.train_state
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0] # jnp.mean does not work on lists
    }

    print('train epoch: %d, loss %.4f, accuracy: %.2f' % (epoch, epoch_metrics_np['loss'],
                                                          epoch_metrics_np['accuracy']*100))

    if cfg.hyperparameters.summary == True:
        train_log_dir = 'logs/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        with train_summary_writer.as_default():
            tf.summary.scalar('accuracy', epoch_metrics_np['accuracy'], step=epoch)

    return state


def eval_model(state, split_ds):
    """

    Parameters
    ----------
    state : train_state.TrainState
        The training state of the experiment.
    split_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to some evaluation set.
    rng : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the random initialization of the neural network.

    Returns
    -------
    """

    logits = eval_step(state, split_ds)

    return logits, split_ds['label']


def eval_model_batched(state, split_ds, rng):
    """

    Parameters
    ----------
    state : train_state.TrainState
        The training state of the experiment.
    split_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to some evaluation set.
    rng : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the random initialization of the neural network.

    Returns
    -------
    """
    split_ds_size = len(split_ds['image'])
    steps = split_ds_size // 500

    perms = jax.random.permutation(rng, split_ds_size)
    perms = perms[:steps*500]
    perms = perms.reshape((steps, 500))
    logits_batch = []
    labels_batch = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in split_ds.items()}
        logits_batch.append(eval_step(state, batch))
        labels_batch.append(batch['label'])

    return jnp.concatenate(logits_batch), jnp.concatenate(labels_batch)


def train_network(cfg: DictConfig):
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
    if cfg.hyperparameters.in_distribution:
        train_ds, test_ds, unlabeled_ds, validation_ds = get_datasets(cfg)
    elif not cfg.hyperparameters.in_distribution:
        a=1

    rng = jax.random.PRNGKey(cfg.hyperparameters.prngkeyseed)#0
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng,cfg)
    del init_rng

    num_epochs = cfg.hyperparameters.epochs

    if cfg.hyperparameters.summary == True:
        test_log_dir = 'logs/test'
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(1, num_epochs+1):
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        if cfg.hyperparameters.mode == 'standard':
            state = train_epoch_standard(state, train_ds, epoch, input_rng, cfg)
        elif cfg.hyperparameters.mode == 'diverse':
            state = train_epoch_diverse(state, train_ds, unlabeled_ds, epoch, input_rng, cfg)

        # Evaluate on the validation set after each training epoch
        logits_total, labels_total = eval_model(state, validation_ds)
        validation_metrics = compute_metrics_jitable(logits=logits_total, labels=labels_total)
        validation_metrics = jax.device_get(validation_metrics)

        print('validation epoch batched: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, validation_metrics['loss'],
            validation_metrics['accuracy']*100))

        if cfg.hyperparameters.summary == True:
            with test_summary_writer.as_default():
                tf.summary.scalar('accuracy', validation_metrics['accuracy'], step=epoch)

    # Evaluate on the validation set after each training epoch
    logits_total, labels_total = eval_model(state, validation_ds)
    validation_metrics = compute_metrics(logits=softmax(logits_total), labels=labels_total)
    validation_metrics = jax.device_get(validation_metrics)

    print('final validation epoch batched: %d, loss: %.2f, accuracy: %.2f, ECE: %.2f, TACE: %.2f, Brier: %.2f' % (
        epoch, validation_metrics['loss'],
        validation_metrics['accuracy']*100,
        validation_metrics['ece'],
        validation_metrics['tace'],
        validation_metrics['brier']))

    os.mkdir(cfg.hyperparameters.CKPT_DIR)
    checkpoints.save_checkpoint(ckpt_dir=cfg.hyperparameters.CKPT_DIR, target=state, step=0)

    if isinstance((1-validation_metrics['accuracy'])**2+validation_metrics['tace']**2, float):
        return (1-validation_metrics['accuracy'])**2+validation_metrics['tace']**2
    else:
        return 10000


def evaluate_saved_models(paths_to_models, split_ds, cfg):
    """
    Gets as an input a list of paths to different models minima. If the list has len>1 then it computes an average of
    the logits treating the list as an ensemble. The metrics are estimated for the average logits (the average is taken
    after the softmax).

    Parameters
    ----------
    paths_to_models : str
        The paths to the folders containing different models.
    split_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to some evaluation set.
    cfg : DictConfig
        The configuration file for the experiment.

    Returns
    -------
    """

    rng = random.PRNGKey(0)

    if isinstance(paths_to_models, str):
        paths_to_models = [paths_to_models]

    preds = []
    for path in paths_to_models:

        #Create dummy state
        state = create_train_state(rng, cfg)

        #Restore state
        restored_state = checkpoints.restore_checkpoint(ckpt_dir=path,
                                                        target=state)
        logits_total, labels_total = eval_model(restored_state, split_ds)
        preds.append(softmax(logits_total))

    validation_metrics = compute_metrics(logits=jnp.mean(jnp.stack(preds), axis=0), labels=labels_total)
    validation_metrics = jax.device_get(validation_metrics)

    return validation_metrics


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
        paths.append(path_to_ensemble + dir + '/ckpts/checkpoint_0')

    validation_metrics = evaluate_saved_models(paths, split_ds=split_ds, cfg=cfg)

    return validation_metrics
