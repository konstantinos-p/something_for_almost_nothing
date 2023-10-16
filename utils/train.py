from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state, checkpoints
import numpy as np
import optax
import tensorflow as tf
from typing import Any
from utils import models
import os
from utils.metrics import get_ece, get_tace, get_brier, mutual_information, get_brier_decomposition
from flax.linen.activation import softmax
import chex
import math
from functools import partial
from jax import jit
from utils.datasets import get_simple_ood_datasets, get_datasets, dataset_num_classes, dataset_dimensions
from utils.datasets import full_random_crop_function, full_random_flip_function
from utils.datasets import get_randomized_datasets, get_canny_sobel_and_original_datasets
import json
from itertools import product

ModuleDef = Any
KeyArray = random.KeyArray
Array = Any
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any
RealNumeric = Any


def cross_entropy_loss(*, logits, labels, num_classes):
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
    labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def cross_entropy_loss_no_softmax(*, logits, labels, num_classes):
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
    labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
    return cross_entropy(logits=logits, labels=labels_onehot).mean()


def cross_entropy(logits: chex.Array, labels: chex.Array,) -> chex.Array:
    """
    Computes the cross entropy between sets of logits and labels.
    """
    chex.assert_type([logits], float)
    return -jnp.log(jnp.sum(labels * logits, axis=-1))


def compute_metrics(*, logits, labels, num_classes):
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
    loss = cross_entropy_loss_no_softmax(logits=logits, labels=labels, num_classes=num_classes)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    ece = get_ece(logits, labels)
    tace = get_tace(logits, labels)
    brier = get_brier(logits, labels)
    brier_uncertainty, brier_resolution, brier_reliability = get_brier_decomposition(logits, labels)

    try:
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'ece': ece.item(),
            'tace': tace.item(),
            'brier': brier.item(),
            'brier_uncertainty': brier_uncertainty.item(),
            'brier_resolution': brier_resolution.item(),
            'brier_reliability': brier_reliability.item()
        }
    except:
        metrics = {
            'loss': 100,
            'accuracy': 0,
            'ece': 1,
            'tace': 1,
            'brier': 100,
            'brier_uncertainty': 1,
            'brier_resolution': 0,
            'brier_reliability': 1
        }

    return metrics


def compute_metrics_jitable(*, logits, labels, num_classes):
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
    loss = cross_entropy_loss(logits=logits, labels=labels, num_classes=num_classes)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }

    return metrics


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

        try:
            weight_decay = cfg.optimizer.weight_decay
        except:
            weight_decay = 0.0001

        tx = optax.adamw(learning_rate=cfg.optimizer.learning_rate, weight_decay=weight_decay)

    return train_state.TrainState.create(apply_fn=network.apply, params=params, tx=tx)


# @jax.jit
@partial(jit, static_argnums=3)
def train_step_standard(state, batch, dropout_rng, num_classes):
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
            return cross_entropy_loss(logits=logits, labels=y, num_classes=num_classes), logits
        total_loss, total_logits = jax.vmap(per_sample_loss_and_logits)(batch_x, batch_y)
        return jnp.mean(total_loss, axis=0), total_logits

    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, logits = grad_fn(state.params, batch['image'], batch['label'])
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics_jitable(logits=logits, labels=batch['label'], num_classes=num_classes)
    return state, metrics, new_dropout_rng


@partial(jit, static_argnames=['num_classes'])
def train_step_nu_ensemble(state,
                           batch_training,
                           batch_unlabeled,
                           dropout_rng,
                           beta,
                           num_classes,
                           ):
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
    beta: float
        The strength of the diversity regularization.
    num_classes: int
        The number of classes in the classification problem.

    Returns
    -------
    state : train_state.TrainState
        The training state of the experiment.
    metrics : dict
        A python dictionary with keys "loss" and "accuracy" corresponding to the cross-entropy loss and the accuracy
        for some logits and labels.
    metrics_unlabeled : dict
        A python dictionary with keys "loss" and "accuracy" corresponding to the cross-entropy loss and the accuracy
        for some logits and labels for the unlabeled set.
    new_dropout_rng : jax.random.PRNGKey
        New pseudo-random number generator (PRNG) key for the randomness of the dropout layers.

    """
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params, batch_x, batch_y):
        def per_sample_loss_and_logits(x, y):
            logits = state.apply_fn(params, x, train=True, rngs={'dropout': dropout_rng})
            return cross_entropy_loss(logits=logits, labels=y, num_classes=num_classes), logits
        total_loss, total_logits = jax.vmap(per_sample_loss_and_logits)(batch_x, batch_y)
        return jnp.mean(total_loss, axis=0), total_logits

    grad_fn = jax.grad(loss_fn, has_aux=True)

    # Compute gradients and logits for training set
    grads_training, logits_training = grad_fn(state.params, batch_training['image'], batch_training['label'])

    # Compute gradients and logits for unlabeled set
    grads_unlabeled, logits_unlabeled = grad_fn(state.params, batch_unlabeled['image'], batch_unlabeled['label'])

    # Compute the complete gradients by weighting the gradients for the unlabeled set using beta
    grads_total = jax.tree_map(lambda x, y: x + beta*y, grads_training, grads_unlabeled)

    state = state.apply_gradients(grads=grads_total)

    metrics_train = compute_metrics_jitable(logits=logits_training,
                                            labels=batch_training['label'],
                                            num_classes=num_classes)

    metrics_unlabeled = compute_metrics_jitable(logits=logits_unlabeled,
                                                labels=batch_unlabeled['label'],
                                                num_classes=num_classes)

    return state, metrics_train, metrics_unlabeled, new_dropout_rng


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


def train_epoch_diverse(state, train_ds, unlabeled_ds, epoch, rng, cfg, random_key_seed):
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
    if cfg.hyperparameters.unlabeled_batch_size == 'same':

        train_ds_size = len(train_ds['image'])
        unlabeled_ds_size = len(unlabeled_ds['image'])
        steps_per_epoch = train_ds_size // cfg.hyperparameters.batch_size_train
        batch_size_unlabeled = cfg.hyperparameters.batch_size_train

        perms_train = jax.random.permutation(rng, train_ds_size)
        perms_unlabeled = jax.random.permutation(rng, unlabeled_ds_size)
        perms_train = perms_train[:steps_per_epoch*cfg.hyperparameters.batch_size_train]
        perms_train = perms_train.reshape((steps_per_epoch, cfg.hyperparameters.batch_size_train))

        if steps_per_epoch * batch_size_unlabeled <= perms_unlabeled.shape[0]:
            perms_unlabeled = perms_unlabeled[:steps_per_epoch * batch_size_unlabeled]
        else:
            fit = int((steps_per_epoch * batch_size_unlabeled) / perms_unlabeled.shape[0])
            rem = ((steps_per_epoch * batch_size_unlabeled) % perms_unlabeled.shape[0])
            perms_unlabeled = jnp.concatenate(fit * [perms_unlabeled])
            perms_unlabeled = jnp.concatenate([perms_unlabeled, perms_unlabeled[0:rem]])
        perms_unlabeled = perms_unlabeled.reshape((steps_per_epoch, batch_size_unlabeled))

    elif cfg.hyperparameters.unlabeled_batch_size == 'full':

        train_ds_size = len(train_ds['image'])
        unlabeled_ds_size = len(unlabeled_ds['image'])
        steps_per_epoch = unlabeled_ds_size // cfg.hyperparameters.batch_size_train

        rng, rng_in = jax.random.split(rng)
        perms_train = jax.random.permutation(rng_in, train_ds_size)
        rng, rng_in = jax.random.split(rng)
        perms_unlabeled = jax.random.permutation(rng_in, unlabeled_ds_size)

        perms_unlabeled = perms_unlabeled[:steps_per_epoch*cfg.hyperparameters.batch_size_train]
        perms_unlabeled = perms_unlabeled.reshape((steps_per_epoch, cfg.hyperparameters.batch_size_train))

        if steps_per_epoch * cfg.hyperparameters.batch_size_train <= perms_train.shape[0]:
            perms_train = perms_train[:steps_per_epoch * cfg.hyperparameters.batch_size_train]
        else:
            fit = int((steps_per_epoch * cfg.hyperparameters.batch_size_train) / perms_train.shape[0])
            rem = ((steps_per_epoch * cfg.hyperparameters.batch_size_train) % perms_train.shape[0])
            perms_train = jnp.concatenate(fit * [perms_train])
            perms_train = jnp.concatenate([perms_train, perms_train[0:rem]])
        perms_train = perms_train.reshape((steps_per_epoch, cfg.hyperparameters.batch_size_train))

    batch_metrics = []
    batch_metrics_unlabeled = []
    dropout_rng, other_key = jax.random.split(rng)
    for perm_both in zip(perms_train, perms_unlabeled):
        batch_train = {k: v[perm_both[0], ...] for k, v in train_ds.items()}
        try:
            if cfg.hyperparameters.augmentations:
                aug_key, other_key = jax.random.split(other_key)
                batch_train['image'] = full_random_flip_function(batch_train['image'], aug_key)
                if cfg.hyperparameters.dataset_name == 'Cifar10' or cfg.hyperparameters.dataset_name == 'Cifar100':
                    aug_key, other_key = jax.random.split(other_key)
                    batch_train['image'] = full_random_crop_function(batch_train['image'], aug_key)
        except:
            do_nothing = 1
        batch_unlabeled = {k: v[perm_both[1], ...] for k, v in unlabeled_ds.items()}

        try:
            train_step_other_methods = cfg.hyperparameters.train_step_other_methods
        except:
            train_step_other_methods = None

        if not train_step_other_methods is None:
            state, metrics, metrics_unlabeled, dropout_rng = train_step_nu_ensemble(state,
                                                                 batch_train,
                                                                 batch_unlabeled,
                                                                 dropout_rng,
                                                                 cfg.hyperparameters.beta,
                                                                 dataset_num_classes[cfg.hyperparameters.dataset_name])

        batch_metrics.append(metrics)
        if train_step_other_methods == 'nu_ensemble':
            batch_metrics_unlabeled.append(metrics_unlabeled)


    # compute mean of metrics across each batch in epoch.train_state
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0] # jnp.mean does not work on lists
    }
    print('train epoch: %d, loss %.4f, accuracy: %.2f' % (epoch, epoch_metrics_np['loss'],
                                                          epoch_metrics_np['accuracy']*100))

    if train_step_other_methods == 'nu_ensemble':
        batch_metrics_unlabeled_np = jax.device_get(batch_metrics_unlabeled)
        epoch_metrics_unlabeled_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_unlabeled_np])
            for k in batch_metrics_unlabeled_np[0]  # jnp.mean does not work on lists
        }
        print('train epoch unlabeled data: %d, loss %.4f, accuracy: %.2f' % (epoch, epoch_metrics_unlabeled_np['loss'],
                                                              epoch_metrics_unlabeled_np['accuracy'] * 100))


    if cfg.hyperparameters.summary == True:
        train_log_dir = str(random_key_seed)+'/logs/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        with train_summary_writer.as_default():
            tf.summary.scalar('accuracy', epoch_metrics_np['accuracy'], step=epoch)

    return state


def train_epoch_standard(state, train_ds, epoch, rng, cfg, random_key_seed):
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
    dropout_rng, other_key = jax.random.split(rng)
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        try:
            if cfg.hyperparameters.augmentations:
                aug_key, other_key = jax.random.split(other_key)
                batch['image'] = full_random_flip_function(batch['image'], aug_key)
                if cfg.hyperparameters.dataset_name == 'Cifar10' or cfg.hyperparameters.dataset_name == 'Cifar100':
                    aug_key, other_key = jax.random.split(other_key)
                    batch['image'] = full_random_crop_function(batch['image'], aug_key)
        except:
            do_nothing =1
        state, metrics, dropout_rng = train_step_standard(state, batch, dropout_rng,
                                                          dataset_num_classes[cfg.hyperparameters.dataset_name])
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
        train_log_dir = str(random_key_seed)+'/logs/train'
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


def train_network(cfg: DictConfig, train_ds, validation_ds, unlabeled_ds, random_key_seed=0):
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

    rng = jax.random.PRNGKey(random_key_seed)#0
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng, cfg)
    del init_rng

    num_epochs = cfg.hyperparameters.epochs

    if cfg.hyperparameters.summary == True:
        test_log_dir = str(random_key_seed)+'/logs/test'
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(1, num_epochs+1):
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        if cfg.hyperparameters.mode == 'standard':
            state = train_epoch_standard(state, train_ds, epoch, input_rng, cfg, random_key_seed)
        elif cfg.hyperparameters.mode == 'diverse':
            state = train_epoch_diverse(state, train_ds, unlabeled_ds, epoch, input_rng, cfg, random_key_seed)

        # Evaluate on the validation set after each training epoch
        logits_total, labels_total = eval_model(state, validation_ds)
        validation_metrics = compute_metrics_jitable(logits=logits_total, labels=labels_total,
                                                     num_classes=dataset_num_classes[cfg.hyperparameters.dataset_name])
        validation_metrics = jax.device_get(validation_metrics)

        print('validation epoch batched: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, validation_metrics['loss'],
            validation_metrics['accuracy']*100))

        if cfg.hyperparameters.summary == True:
            with test_summary_writer.as_default():
                tf.summary.scalar('accuracy', validation_metrics['accuracy'], step=epoch)

    # Evaluate on the validation set after each training epoch
    logits_total, labels_total = eval_model(state, validation_ds)
    validation_metrics = compute_metrics(logits=softmax(logits_total), labels=labels_total,
                                         num_classes=dataset_num_classes[cfg.hyperparameters.dataset_name])
    validation_metrics = jax.device_get(validation_metrics)

    print('final validation epoch batched: %d, loss: %.2f, accuracy: %.2f, ECE: %.2f, TACE: %.2f, Brier: %.2f' % (
        epoch, validation_metrics['loss'],
        validation_metrics['accuracy']*100,
        validation_metrics['ece'],
        validation_metrics['tace'],
        validation_metrics['brier']))

    os.mkdir(str(random_key_seed))
    os.mkdir(str(random_key_seed)+'/'+cfg.hyperparameters.CKPT_DIR)
    checkpoints.save_checkpoint(ckpt_dir=str(random_key_seed)+'/'+cfg.hyperparameters.CKPT_DIR, target=state, step=0)

    if isinstance((1-validation_metrics['accuracy'])**2+validation_metrics['tace']**2, float):
        return (1-validation_metrics['accuracy'])**2+validation_metrics['tace']**2
    else:
        return 10000


def train_ensemble(cfg: DictConfig):

    try:
        different_datasets_per_ensemble_member = cfg.hyperparameters.different_datasets_per_ensemble_member
    except:
        different_datasets_per_ensemble_member = False

    if cfg.hyperparameters.dataset_name == 'Cifar10_dominoes' or \
            cfg.hyperparameters.dataset_name =='fashion_mnist_dominoes':
        train_ds, test_ds, unlabeled_ds, validation_ds = get_simple_ood_datasets(cfg)
    elif cfg.hyperparameters.dataset_name == 'Cifar10' or \
            cfg.hyperparameters.dataset_name =='fashion_mnist' or \
            cfg.hyperparameters.dataset_name == 'Cifar100' or \
            cfg.hyperparameters.dataset_name =='svhn_cropped':

        if different_datasets_per_ensemble_member:
            if cfg.hyperparameters.dataset_individualization == 'randomized':
                train_ds, test_ds, unlabeled_ds, validation_ds = get_randomized_datasets(cfg)
            elif cfg.hyperparameters.dataset_individualization == 'canny_sobel':
                train_ds, test_ds, unlabeled_ds, validation_ds = get_canny_sobel_and_original_datasets(cfg)
        else:
            train_ds, test_ds, unlabeled_ds, validation_ds = get_datasets(cfg)

    for random_key_seed in range(cfg.hyperparameters.ensemble_size):

        if not different_datasets_per_ensemble_member:

            train_network(cfg=cfg,
                          train_ds=train_ds,
                          validation_ds=validation_ds,
                          unlabeled_ds=unlabeled_ds,
                          random_key_seed=random_key_seed)

        else:

            train_network(cfg=cfg,
                          train_ds=train_ds[random_key_seed],
                          validation_ds=validation_ds,
                          unlabeled_ds=unlabeled_ds[random_key_seed],
                          random_key_seed=random_key_seed)

    test_metrics = evaluate_ensemble(path_to_ensemble=os.getcwd(),
                                           split_ds=test_ds,
                                           cfg=cfg,
                                           activate_delete=False)

    validation_metrics = evaluate_ensemble(path_to_ensemble=os.getcwd(),
                                           split_ds=validation_ds,
                                           cfg=cfg,
                                           activate_delete=True)

    #Save the metrics
    with open('test_metrics.json', 'w') as outfile:
        json.dump(test_metrics, outfile)
    with open('validation_metrics.json', 'w') as outfile:
        json.dump(validation_metrics, outfile)

    print('Test metrics are: ')
    print(test_metrics)

    print('Validation metrics are: ')
    print(validation_metrics)

    if math.isnan((1 - validation_metrics['accuracy']) ** 2 + validation_metrics['tace'] ** 2):
        return 10000
    else:
        return (1 - validation_metrics['accuracy']) ** 2 + validation_metrics['tace'] ** 2


def evaluate_saved_models(paths_to_models, split_ds, cfg, activate_delete=True):
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

        if cfg.hyperparameters.delete_checkpoints and activate_delete:
            os.remove(path)

    final_preds = jnp.mean(jnp.stack(preds), axis=0)

    validation_metrics = compute_metrics(logits=final_preds, labels=labels_total,
                                         num_classes=dataset_num_classes[cfg.hyperparameters.dataset_name])
    validation_metrics = jax.device_get(validation_metrics)

    # Mutual information between ensemble member predictions
    cat_vars = []
    for pred in preds:
        cat_vars.append(jnp.argmax(pred, -1))

    combinations = product(np.arange(len(cat_vars)), np.arange(len(cat_vars)))

    mutual_info = 0
    count = 0
    for combination in combinations:
        point1 = list(combination)[0]
        point2 = list(combination)[1]
        if point1 != point2:
            mutual_info += mutual_information(np.array(cat_vars[point1]), np.array(cat_vars[point2]))
            count += 1

    validation_metrics['mutual_information_between_predictions'] = mutual_info/count

    return validation_metrics


def evaluate_ensemble(path_to_ensemble, split_ds, cfg, activate_delete=True, ensemble_size=0):
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
            paths.append(path_to_ensemble + '/' + dir + '/ckpts/checkpoint_0')

    if ensemble_size > 0:
        paths = paths[:ensemble_size]

    validation_metrics = evaluate_saved_models(paths, split_ds=split_ds, cfg=cfg, activate_delete=activate_delete)

    return validation_metrics
