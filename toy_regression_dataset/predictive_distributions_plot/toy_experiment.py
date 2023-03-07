from flax import linen as nn
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
import tensorflow as tf
from typing import Any
import hydra
from flax.training import train_state, checkpoints
import os

ModuleDef = Any
KeyArray = random.KeyArray
Array = Any
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any
RealNumeric = Any



def create_dataset():
    key = random.PRNGKey(0)
    std = 0.5
    start = 0
    end = 10
    sampling = 100

    #Create training, test and unlabeled set
    x = jnp.linspace(start, end, sampling)
    y = jnp.sin(x)+std*random.normal(key=key, shape=x.shape)

    x_training = x[jnp.where(jnp.logical_or(x>6,x<2))]
    y_training = y[jnp.where(jnp.logical_or(x>6,x<2))]

    x_test = x[jnp.where(jnp.logical_and(x<6,x>2))]
    y_test = y[jnp.where(jnp.logical_and(x<6,x>2))]

    key, new_key = random.split(key)
    x_unlabeled = jnp.linspace(start, end, sampling)
    y_unlabeled = jnp.sin(x)+std*random.normal(key=new_key, shape=x.shape)

    # Expand dimensions
    x_training = jnp.expand_dims(x_training, axis=1)
    y_training = jnp.expand_dims(y_training, axis=1)

    y_test = jnp.expand_dims(y_test, axis=1)
    x_test = jnp.expand_dims(x_test, axis=1)

    y_unlabeled = jnp.expand_dims(y_unlabeled, axis=1)
    x_unlabeled = jnp.expand_dims(x_unlabeled, axis=1)
    return x_training, y_training, y_test, x_test, x_unlabeled, y_unlabeled

x_training, y_training, y_test, x_test, x_unlabeled, y_unlabeled = create_dataset()

class MLP(nn.Module):
    """
    A simple MLP model.
    """

    @nn.compact
    def __call__(self, x, train):
        out = nn.Dense(features=100)(x)
        out = nn.relu(out)
        out = nn.Dense(features=100)(out)
        out = nn.relu(out)
        out = nn.Dense(features=100)(out)
        out = nn.relu(out)
        out = nn.Dense(features=1)(out)
        return out


def squared_error(*, logits, targets):
    """
    The squared error. Can also be seen as NLL with a Gaussian likelihood.
    Parameters
    ----------
    logits: float
        The prediction preactivations of the neural network.
    targets:
        The regression targets.

    Returns
    -------
        : optax.l2_loss
        The l2 loss.

    """
    return optax.l2_loss(predictions=logits, targets=targets).mean()


def compute_metrics(*, logits, targets):
    """
    Computes the crossentropy loss and the accuracy for a given set of predictions and groundtruth labels.
    Parameters
    ----------
    logits: float
        The prediction preactivations of the neural network.
    labels:
        The groundtruth labels.

    Returns
    -------
    metrics: dict
        A python dictionary with keys "loss" and "accuracy" corresponding to the cross-entropy loss and the accuracy
        for some logits and labels.

    """
    loss = squared_error(logits=logits, targets=targets)
    metrics = {
        'loss': loss
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
    network = MLP()
    params = network.init(rng, jnp.ones([1, 1]), train=False)
    tx = optax.adamw(cfg.hyperparameters.learning_rate)
    return train_state.TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def train_epoch(state, epoch, rng, cfg):
    """
    Train for a single epoch.
    Parameters
    ----------
    state : train_state.TrainState
        The training state of the experiment.
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

    batch_metrics = []
    dropout_rng = jax.random.split(rng, jax.local_device_count())[0]
    if cfg.hyperparameters.mode == 'standard':
        state, metrics, dropout_rng = train_step(state, dropout_rng)
    elif cfg.hyperparameters.mode == 'diverse':
        state, metrics, dropout_rng = train_step_diverse(state, dropout_rng, cfg.hyperparameters.mylambda,
                                                         cfg.hyperparameters.prior_var,
                                                         cfg.hyperparameters.beta)

    batch_metrics.append(metrics)

    #compute mean of metrics across each batch in epoch.train_state
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0] # jnp.mean does not work on lists
    }

    print('train epoch: %d, loss %.4f' % (epoch, epoch_metrics_np['loss']))

    if epoch % 100 == 0:
        train_log_dir = 'logs/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        with train_summary_writer.as_default():
            tf.summary.scalar('train_l2_error', epoch_metrics_np['loss'], step=epoch)

    return state


def eval_model(state):
    """
    Parameters
    ----------
    state : train_state.TrainState
        The training state of the experiment.
    test_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the test set.

    Returns
    -------
    """

    metrics = eval_step(state)
    metrics = jax.device_get(metrics)
    summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
    return summary['loss']


@jax.jit
def evaluate_model_outputs(state, input):
    """
    A single evaluation step of the output logits of the neural network for a batch of inputs, as well as the
    cross-entropy loss and the classification accuracy.
    Parameters
    ----------
    state : train_state.TrainState
        The training state of the experiment.

    Returns
    -------
    logits : jnp.array
        the outputs of the neural network for a given input
    """
    logit_outputs = state.apply_fn(state.params, input, train=False)

    return logit_outputs


@jax.jit
def eval_step(state):
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
    logits = state.apply_fn(state.params, x_test, train=False)
    return compute_metrics(logits=logits, targets=y_test)


@jax.jit
def train_step(state, dropout_rng):
    """
    Trains the neural network for a single step."
    Parameters
    ----------
    state : train_state.TrainState
        The initial training state of the experiment.
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

    def loss_fn(params):
        logits = state.apply_fn(params, x_training, train=True, rngs={'dropout': dropout_rng})
        loss = squared_error(logits=logits, targets=y_training)
        return loss, logits
    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, logits = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, targets=y_training)
    return state, metrics, new_dropout_rng


@jax.jit
def train_step_diverse(state, dropout_rng, mylambda, prior_var, beta):
    """
    Trains the neural network for a single step."
    Parameters
    ----------
    state : train_state.TrainState
        The initial training state of the experiment.
    dropout_rng : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the randomness of the dropout layers.
    mylambda: float
    prior_var : float
    beta: float



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

    def loss_fn(params):
        logits = state.apply_fn(params, x_training, train=True, rngs={'dropout': dropout_rng})
        loss = squared_error(logits=logits, targets=y_training)
        return loss, logits

    def output_logits_training(params, x_training):
        logits = state.apply_fn(params, x_training, train=True, rngs={'dropout': dropout_rng})
        logits_sum = jnp.sum(logits)
        return logits_sum

    def output_logits_unlabeled(params, x_unlabeled):
        logits = state.apply_fn(params, x_unlabeled, train=True, rngs={'dropout': dropout_rng})
        logits_sum = jnp.sum(logits)
        return logits_sum

    grad_fn = jax.grad(loss_fn, has_aux=True)
    output_training_grad_fn = jax.vmap(jax.grad(output_logits_training, has_aux=False), in_axes=(None, 0))
    output_unlabeled_grad_fn = jax.vmap(jax.grad(output_logits_unlabeled, has_aux=False), in_axes=(None, 0))

    grads, logits = grad_fn(state.params)
    grads_diverse_training = output_training_grad_fn(state.params, x_training)
    grads_diverse_unlabeled = output_unlabeled_grad_fn(state.params, x_unlabeled)

    def diverse_term(grads_diverse_training, grads_diverse_unlabeled, mylambda, prior_var):

        term1_pow2 = jax.tree_map(lambda x: jnp.sum(x**2, axis=0), grads_diverse_training)
        term1_pow3 = jax.tree_map(lambda x: jnp.sum(x ** 3, axis=0), grads_diverse_training)
        term1 = jax.tree_map(lambda x, y: -(mylambda*x+1/prior_var)**(-2)*(2*mylambda*y), term1_pow2, term1_pow3)

        term2 = jax.tree_map(lambda x: jnp.sum(x**2, axis=0), grads_diverse_unlabeled)

        term3 = jax.tree_map(lambda x: (mylambda*x+1/prior_var)**(-1), term1_pow2)

        term4 = jax.tree_map(lambda x: jnp.sum(2*x**3, axis=0), grads_diverse_unlabeled)

        return jax.tree_map(lambda x, y, z, k: x*y+z*k, term1, term2, term3, term4)

    divers_grads = jax.tree_map(lambda x: x*beta, diverse_term(grads_diverse_training, grads_diverse_unlabeled,
                                                               mylambda, prior_var))
    '''
    Note that the diverse gradients that we compute should be subtracted from the normal ones.
    '''
    grads_total = jax.tree_map(lambda x, y: x-y, grads, divers_grads)

    state = state.apply_gradients(grads=grads_total)
    metrics = compute_metrics(logits=logits, targets=y_training)
    return state, metrics, new_dropout_rng


@hydra.main(version_base=None, config_path="conf", config_name="toy_example_train")
def train_network(cfg : DictConfig):
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

    rng = jax.random.PRNGKey(cfg.hyperparameters.prngkeyseed)#0
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng, cfg)
    del init_rng #Must not be used anymore

    num_epochs = cfg.hyperparameters.epochs

    test_log_dir = 'logs/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(1,num_epochs+1):
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        state = train_epoch(state, epoch, input_rng, cfg)
        # Evaluate on the test set after each training epoch
        test_loss = eval_model(state)
        print('test epoch: %d, loss: %.2f' % (epoch, test_loss))

        if epoch % 100 == 0:
            with test_summary_writer.as_default():
                tf.summary.scalar('test_l2_error', test_loss, step=epoch)

    os.mkdir(cfg.hyperparameters.CKPT_DIR)
    checkpoints.save_checkpoint(ckpt_dir=cfg.hyperparameters.CKPT_DIR, target=state, step=epoch)

    return test_loss


if __name__ == '__main__':
    train_network()