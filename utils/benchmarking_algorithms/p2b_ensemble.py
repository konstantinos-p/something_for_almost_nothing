from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import tensorflow as tf
import os
from functools import partial
from jax import jit
from utils.train import create_train_state, cross_entropy_loss, evaluate_ensemble
from utils.datasets import dataset_num_classes, get_datasets


def train_p2b_ensemble(cfg: DictConfig):
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

    train_ds, test_ds, unlabeled_ds, validation_ds = get_datasets(cfg)
    rng = jax.random.PRNGKey(0)#0
    rng, init_rng = jax.random.split(rng)
    directory = os.getcwd()
    states = []
    for i in range(cfg.hyperparameters.ensemble_size):
        states.append(create_train_state(init_rng, cfg))
        rng_dummy, init_rng = jax.random.split(init_rng)

    num_epochs = cfg.hyperparameters.epochs
    if cfg.hyperparameters.summary:
        validation_log_dir = '/logs/validation'
        validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)

    for epoch in range(1, num_epochs+1):
        # Use a separate PRNG key to permute image data during shuffling
        rngs = []
        for i in range(cfg.hyperparameters.ensemble_size):
            rng, input_rng = jax.random.split(rng)
            rngs.append(input_rng)
        states = train_epoch_p2b(states, train_ds, rngs, cfg)
        '''
        # Evaluate on the validation set after each training epoch
        validation_metrics = evaluate_ensemble(directory, validation_ds, cfg)
        validation_metrics = jax.device_get(validation_metrics)

        print('validation epoch batched: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, validation_metrics['loss'],
            validation_metrics['accuracy']*100))

        if cfg.hyperparameters.summary:
            with validation_summary_writer.as_default():
                tf.summary.scalar('accuracy', validation_metrics['accuracy'], step=epoch)
        '''
    validation_metrics = evaluate_ensemble(directory, validation_ds, cfg)
    validation_metrics = jax.device_get(validation_metrics)
    print('final validation epoch batched: %d, loss: %.2f, accuracy: %.2f, ECE: %.2f, TACE: %.2f, Brier: %.2f' % (
        epoch, validation_metrics['loss'],
        validation_metrics['accuracy']*100,
        validation_metrics['ece'],
        validation_metrics['tace'],
        validation_metrics['brier']))

    if isinstance((1-validation_metrics['accuracy'])**2+validation_metrics['tace']**2, float):
        return (1-validation_metrics['accuracy'])**2+validation_metrics['tace']**2
    else:
        return 10000


def train_epoch_p2b(states, train_ds, rngs, cfg):
    """
    Train for a single epoch.
    Parameters
    ----------
    states : train_state.TrainState
        The training state of the experiment.
    train_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the training set.
    rngs : jax.random.PRNGKey
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

    perms = jax.random.permutation(rngs[0], train_ds_size)
    perms = perms[:steps_per_epoch*cfg.hyperparameters.batch_size_train]
    perms = perms.reshape((steps_per_epoch, cfg.hyperparameters.batch_size_train))
    dropout_rngs = []
    for rng in rngs:
        dummy_rng, dropout_rng = jax.random.split(rng)
        dropout_rngs.append(dropout_rng)
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        states, dropout_rngs = train_step_p2b(states,
                                              batch,
                                              dropout_rngs,
                                              dataset_num_classes[cfg.hyperparameters.dataset_name],
                                              cfg.hyperparameters.mylambda)

    for i, state in zip(range(cfg.hyperparameters.ensemble_size), states):
        if not os.path.exists(str(i)+'/'+cfg.hyperparameters.CKPT_DIR):
            os.mkdir(str(i))
            os.mkdir(str(i)+'/'+cfg.hyperparameters.CKPT_DIR)
        else:
            if os.path.exists(str(i)+'/'+cfg.hyperparameters.CKPT_DIR+'/checkpoint_0'):
                os.remove(str(i)+'/'+cfg.hyperparameters.CKPT_DIR+'/checkpoint_0')
        checkpoints.save_checkpoint(ckpt_dir=str(i)+'/'+cfg.hyperparameters.CKPT_DIR, target=state, step=0)

    return states


@partial(jit, static_argnames=['num_classes', 'mylambda'])
def train_step_p2b(states, batch, dropout_rngs, num_classes, mylambda):
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
    new_dropout_rngs = []
    for dropout_rng in dropout_rngs:
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        new_dropout_rngs.append(new_dropout_rng)

    def per_sample_loss_and_logits(params, state, dropout_rng, batch_x, batch_y):
        def vmapped(x, y):
            logits = state.apply_fn(params, x, train=True, rngs={'dropout': dropout_rng})
            return cross_entropy_loss(logits=logits, labels=y, num_classes=num_classes)
        return jax.vmap(vmapped)(batch_x, batch_y)

    def loss_fn(params_list, batch_x, batch_y):
        losses = []
        for params, state, dropout_rng in zip(params_list, states, dropout_rngs):
            losses.append(per_sample_loss_and_logits(params, state, dropout_rng, batch_x, batch_y))
        return losses

    def total_loss(params_list, batch_x, batch_y):
        #total loss
        losses = loss_fn(params_list, batch_x, batch_y)
        losses_stacked = jnp.stack(losses)
        loss_total = jnp.mean(losses_stacked, (0, 1))

        #term 1
        max = jnp.max(losses_stacked, axis=0)
        term1_total = jnp.mean(jnp.exp(2*(-1)*losses_stacked-2*jax.lax.stop_gradient(max)), (0, 1))

        #terms 2
        indexes = jnp.meshgrid(jnp.arange(losses_stacked.shape[0]), jnp.arange(losses_stacked.shape[0]))
        indexes = jnp.array(indexes).T.reshape(-1, 2)
        term2_total = 0
        for i in range(len(indexes)):
            term2_total += jnp.sum(jnp.exp((-1)*losses_stacked[indexes[i][0], :]+(-1)*losses_stacked[indexes[i][1], :]-2*jax.lax.stop_gradient(max)), axis=0)
        term2_total = term2_total/(losses_stacked.shape[0]*losses_stacked.shape[1])
        return loss_total+(term1_total+term2_total)*mylambda

    grad_fn = jax.grad(total_loss)
    params = []
    for state in states:
        params.append(state.params)
    grads = grad_fn(params, batch['image'], batch['label'])
    new_states = []
    for state, grad in zip(states, grads):
        new_states.append(state.apply_gradients(grads=grad))

    return new_states, new_dropout_rngs
