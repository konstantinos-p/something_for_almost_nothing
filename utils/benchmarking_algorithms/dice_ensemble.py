from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state, checkpoints
import optax
import tensorflow as tf
import os
from flax.linen.activation import softmax
from jax import jit
from utils.train import cross_entropy_loss, compute_metrics
from utils.datasets import dataset_num_classes, dataset_dimensions, get_datasets
from utils.benchmarking_models import benchmarking_models
from flax import linen as nn


def beta_coef(epoch, cfg):
    thresholds = jnp.array([0, 0.2, 0.58, 0.83])
    values = jnp.logspace(cfg.hyperparameters.beta_start, cfg.hyperparameters.beta_end, 4)
    index = jnp.where(epoch/cfg.hyperparameters.epochs > thresholds)
    res = 1/values[index[0][-1]]
    return res


def delta_coef(epoch, cfg):
    res = cfg.hyperparameters.final_delta*epoch/cfg.hyperparameters.epochs
    return res


def create_dice_train_state(rng, cfg):
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
    model_cls = getattr(benchmarking_models, cfg.hyperparameters.model)
    network = model_cls()
    params_network = network.init(rng, jnp.ones(dataset_dimensions[cfg.hyperparameters.dataset_name]), train=False)

    network_out_features = network.apply(params_network, jnp.ones(dataset_dimensions[cfg.hyperparameters.dataset_name]),
                                         train=True, rngs={'dropout': rng})
    variance_layer = benchmarking_models.VariancePredictor(num_outputs=network_out_features.shape[0])
    params_variance = variance_layer.init(rng, network_out_features, train=False)

    classifier = benchmarking_models.LinearLayer(num_outputs=dataset_num_classes[cfg.hyperparameters.dataset_name])
    params_classifier = classifier.init(rng, network_out_features, train=False)

    backwards_encoder = benchmarking_models.LinearLayer(num_outputs=network_out_features.shape[0])
    params_backwards = backwards_encoder.init(rng, jnp.ones(dataset_num_classes[cfg.hyperparameters.dataset_name]), train=False)

    optimizers = []
    if cfg.optimizer.name == 'sgd':
        for i in range(4):
            optimizers.append(optax.sgd(cfg.optimizer.learning_rate, cfg.optimizer.momentum))
    elif cfg.optimizer.name == 'adamw':
        for i in range(4):
            optimizers.append(optax.adamw(cfg.optimizer.learning_rate))

    network_state = train_state.TrainState.create(apply_fn=network.apply, params=params_network,
                                                  tx=optimizers[0])
    variance_state = train_state.TrainState.create(apply_fn=variance_layer.apply, params=params_variance,
                                                   tx=optimizers[1])
    classifier_state = train_state.TrainState.create(apply_fn=classifier.apply, params=params_classifier,
                                                     tx=optimizers[2])
    backwards_state = train_state.TrainState.create(apply_fn=backwards_encoder.apply, params=params_backwards,
                                                    tx=optimizers[3])

    return {'network': network_state, 'variance': variance_state, 'classifier': classifier_state, 'backwards': backwards_state}, network_out_features.shape[0]


def create_adversarial_state(rng, cfg, features_dim):
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
    model_cls = getattr(benchmarking_models, 'DiceDescriminator')
    descriminator = model_cls()
    params = descriminator.init(rng, jnp.ones(2*features_dim), train=False)
    if cfg.optimizer.name == 'sgd':
        tx = optax.sgd(cfg.optimizer.learning_rate, cfg.optimizer.momentum)
    elif cfg.optimizer.name == 'adamw':
        tx = optax.adamw(cfg.optimizer.learning_rate)
    return train_state.TrainState.create(apply_fn=descriminator.apply, params=params, tx=tx)


def train_dice_ensemble(cfg: DictConfig):
    """
    Function that trains a DICE ensemble.


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
    states_list_dict = []
    for i in range(cfg.hyperparameters.ensemble_size):
        states_dict, features_dim = create_dice_train_state(init_rng, cfg)
        states_list_dict.append(states_dict)
        rng_dummy, init_rng = jax.random.split(init_rng)

    adversarial_state = create_adversarial_state(init_rng, cfg, features_dim)

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
        states_list_dict = train_epoch_dice(states_list_dict, adversarial_state, train_ds, rngs, cfg, epoch, num_epochs)


        # Evaluate on the validation set after each training epoch
        '''
        validation_metrics = evaluate_ensemble(directory, validation_ds, cfg)
        validation_metrics = jax.device_get(validation_metrics)
        print('final validation epoch batched: %d, loss: %.2f, accuracy: %.2f, ECE: %.2f, TACE: %.2f, Brier: %.2f' % (
            epoch, validation_metrics['loss'],
            validation_metrics['accuracy']*100,
            validation_metrics['ece'],
            validation_metrics['tace'],
            validation_metrics['brier']))

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


def train_epoch_dice(states_list_dict, adversarial_state, train_ds, rngs, cfg, epoch, num_epochs):
    """
    Train a DICE ensemble for a single epoch.

    Parameters
    ----------
    states_list_dict : list of dict
        A list of dictionaries with keys 'network', 'classifier', 'variance', 'backwards'. Each element of the
        list corresponds to an ensemble member. The keys give access to a train_state.TrainState for each of the
        architecture components.
    adversarial_state: train_state.TrainState
        The state of the Adversarial discriminator.
    train_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the training set.
    rngs : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the random initialization of the neural network.
    cfg : DictConfig
        The configuration file for the experiment.
    epoch: int
        The current training epoch.
    num_epochs: int
        The total number of epochs.

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
        states_list_dict, adversarial_state, dropout_rngs = vceb_and_diversity_loss_step(
                                                                        states_list_dict,
                                                                        adversarial_state,
                                                                        batch,
                                                                        dropout_rngs,
                                                                        dataset_num_classes[cfg.hyperparameters.dataset_name],
                                                                        epoch,
                                                                        cfg)

    for i, state_dict in zip(range(cfg.hyperparameters.ensemble_size), states_list_dict):
        if not os.path.exists(str(i)+'/'+cfg.hyperparameters.CKPT_DIR):
            os.mkdir(str(i))
            os.mkdir(str(i)+'/'+cfg.hyperparameters.CKPT_DIR)
        else:
            if os.path.exists(str(i)+'/'+cfg.hyperparameters.CKPT_DIR+'/network_0'):
                os.remove(str(i)+'/'+cfg.hyperparameters.CKPT_DIR+'/network_0')
            if os.path.exists(str(i)+'/'+cfg.hyperparameters.CKPT_DIR+'/classifier_0'):
                os.remove(str(i)+'/'+cfg.hyperparameters.CKPT_DIR+'/classifier_0')
        checkpoints.save_checkpoint(ckpt_dir=str(i)+'/'+cfg.hyperparameters.CKPT_DIR, target=state_dict['network'], step=0, prefix='network_')
        checkpoints.save_checkpoint(ckpt_dir=str(i) + '/' + cfg.hyperparameters.CKPT_DIR, target=state_dict['classifier'], step=0, prefix='classifier_')

    return states_list_dict


#@partial(jit, static_argnames=['num_classes'])
def vceb_and_diversity_loss_step(states_list_dict, adversarial_state, batch, dropout_rngs, num_classes, epoch,
                                 cfg):
    """
    Applies a single training step on the DICE ensemble. This is then followed by 4 training steps on the
    Discriminator of the architecture, that promotes feature diversity."

    Parameters
    ----------
    states_list_dict : list of dict
        A list of dictionaries with keys 'network', 'classifier', 'variance', 'backwards'. Each element of the
        list corresponds to an ensemble member. The keys give access to a train_state.TrainState for each of the
        architecture components.
    adversarial_state: train_state.TrainState
        The state of the Adversarial discriminator.
    batch : dict
        Dictionary with keys 'image' and 'label' corresponding to a batch of the training set.
    dropout_rngs : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the randomness of the dropout layers.
    num_classes: int
        The number of classes in the classification problem.
    epoch: int
        The current training epoch.
    num_epochs: int
        The total number of epochs.

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

    #The VCEB loss per sample
    def per_sample_loss_and_features(params_f, params_v, params_c, params_b,
                                   state_f, state_v, state_c, state_b,
                                   dropout_rng, batch_x, batch_y):
        def vmapped(x, y):
            # Classifier
            features = state_f.apply_fn(params_f, x, train=True, rngs={'dropout': dropout_rng})
            std = state_v.apply_fn(params_v, features, train=True, rngs={'dropout': dropout_rng})
            gauss_normal_sample = jax.random.normal(dropout_rng, shape=[std.shape[0]])
            sampled = features+gauss_normal_sample*std
            logits = state_c.apply_fn(params_c, sampled, train=True, rngs={'dropout': dropout_rng})
            # Regulariser
            prototype = state_b.apply_fn(params_b, jax.nn.one_hot(y, num_classes=num_classes), train=True, rngs={'dropout': dropout_rng})

            return cross_entropy_loss(logits=logits, labels=y, num_classes=num_classes)\
                   + beta_coef(epoch, cfg)*optax.l2_loss(prototype, features).sum()\
                   + beta_coef(epoch, cfg)*(1+std**2-jnp.log(std**2)).sum(), sampled
            '''
            return cross_entropy_loss(logits=logits, labels=y, num_classes=num_classes), sampled
            '''
        return jax.vmap(vmapped)(batch_x, batch_y)

    # The VCEB loss as well as the adversarial loss
    def loss_fn(params_list_f, params_list_v, params_list_c, params_list_b, batch_x, batch_y):
        vceb_losses = []
        features_list = []
        for params_f, params_v, params_c, params_b, state, dropout_rng in zip(
                params_list_f, params_list_v, params_list_c, params_list_b, states_list_dict, dropout_rngs):
            vceb, sampled_features = per_sample_loss_and_features(params_f, params_v, params_c, params_b,
                                                                  state['network'], state['variance'], state['classifier'], state['backwards'],
                                                                  dropout_rng, batch_x, batch_y)
            vceb_losses.append(jnp.mean(vceb, axis=0))
            features_list.append(sampled_features)

        adversarial_losses = []

        def adversarial_loss_per_sample(x):
            adversarial = adversarial_state.apply_fn(adversarial_state.params, x, train=False, rngs={'dropout': dropout_rng})
            return 10 * nn.tanh(jnp.log(adversarial/(1-adversarial)) / 10)

        for i in range(len(features_list)):
            for j in range(i, len(features_list)):
                stacked = jnp.concatenate([features_list[i], features_list[j]], axis=1)
                adversarial_losses.append(jax.vmap(adversarial_loss_per_sample)(stacked))

        return jnp.mean(jnp.stack(vceb_losses))+delta_coef(epoch, cfg)*jnp.mean(jnp.stack(adversarial_losses)), \
               features_list
        #return jnp.mean(jnp.stack(vceb_losses)), features_list

    # Applying the gradients to the architecture components
    grad_fn = jax.grad(loss_fn, argnums=[0, 1, 2, 3], has_aux=True)
    params = [[], [], [], []]
    for state in states_list_dict:
        params[0].append(state['network'].params)
        params[1].append(state['variance'].params)
        params[2].append(state['classifier'].params)
        params[3].append(state['backwards'].params)
    a, features_list = grad_fn(params[0], params[1], params[2], params[3], batch['image'], batch['label'])
    grads1 = a[0]
    grads2 = a[1]
    grads3 = a[2]
    grads4 = a[3]
    new_states = []
    for state, grad1, grad2, grad3, grad4 in zip(states_list_dict, grads1, grads2, grads3, grads4):
        state['network'] = state['network'].apply_gradients(grads=grad1)
        state['variance'] = state['variance'].apply_gradients(grads=grad2)
        state['classifier'] = state['classifier'].apply_gradients(grads=grad3)
        state['backwards'] = state['backwards'].apply_gradients(grads=grad4)
        new_states.append(state)

    # Train the Discriminator for 4 steps
    for adv_steps in range(4):
        #Adversarial training
        samples_adversarial = 50
        samples_joint = []
        samples_product = []
        for i in range(samples_adversarial):
            # Joint distribution sample
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
            perm = jax.random.permutation(dropout_rng, jnp.arange(len(states_list_dict)), axis=0, independent=False)
            perm_samples = jax.random.permutation(dropout_rng, jnp.arange(features_list[0].shape[0]), axis=0, independent=False)
            sample_joint = jnp.concatenate([features_list[perm[0]], features_list[perm[1]]], axis=1)[perm_samples[0]]
            samples_joint.append(sample_joint)

            # Product distribution sample
            notfound = True
            while notfound:
                dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
                random_class = jax.random.randint(dropout_rng, shape=[1], minval=0, maxval=num_classes)
                if jnp.where(batch['label'] == random_class)[0].shape[0] > 1:
                    notfound = False
            perm = jax.random.permutation(dropout_rng, jnp.arange(len(states_list_dict)), axis=0, independent=False)
            perm_samples_product = jax.random.permutation(dropout_rng, jnp.where(batch['label'] == random_class)[0], axis=0,
                                                  independent=False)

            sample_product = jnp.concatenate([features_list[perm[0]][perm_samples_product[0]], features_list[perm[1]][perm_samples_product[1]]])
            samples_product.append(sample_product)

        def adversarial_loss(params, joint, product):
            def per_sample_loss_and_logits(joint_1, product_1):
                pred_1 = adversarial_state.apply_fn(params, joint_1, train=True, rngs={'dropout': dropout_rng})
                pred_2 = adversarial_state.apply_fn(params, product_1, train=True, rngs={'dropout': dropout_rng})
                return -jnp.log(pred_1)+jnp.log(1-pred_2)
            total_loss = jax.vmap(per_sample_loss_and_logits)(joint, product)
            return jnp.mean(total_loss, axis=0)[0]

        adversarial_grad_fn = jax.grad(adversarial_loss, has_aux=False)
        adversarial_grads = adversarial_grad_fn(adversarial_state.params, jnp.stack(samples_joint), jnp.stack(samples_product))
        adversarial_state = adversarial_state.apply_gradients(grads=adversarial_grads)

    return new_states, adversarial_state, new_dropout_rngs


def evaluate_saved_models(paths_network, paths_classifier, split_ds, cfg):
    """
    Gets as an input a list of paths to different models minima. If the list has len>1 then it computes an average of
    the logits treating the list as an ensemble. The metrics are estimated for the average logits (the average is taken
    after the softmax).

    Parameters
    ----------
    paths_network : str
        The paths to the folders containing different network parameters.
    paths_classifier : str
        The paths to the folders containing different classifier parameters.
    split_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to some evaluation set.
    cfg : DictConfig
        The configuration file for the experiment.

    Returns
    -------
    """

    rng = random.PRNGKey(0)

    if isinstance(paths_network, str):
        paths_network = [paths_network]
    if isinstance(paths_classifier, str):
        paths_classifier = [paths_classifier]

    preds = []
    for path_network, path_classifier in zip(paths_network, paths_classifier):

        #Create dummy state
        state_dict, features_dim = create_dice_train_state(rng, cfg)

        #Restore state
        restored_network_state = checkpoints.restore_checkpoint(ckpt_dir=path_network,
                                                        target=state_dict['network'])
        restored_classifier_state = checkpoints.restore_checkpoint(ckpt_dir=path_classifier,
                                                        target=state_dict['classifier'])
        logits_total, labels_total = eval_model(restored_network_state, restored_classifier_state, split_ds)
        preds.append(softmax(logits_total))

        if cfg.hyperparameters.delete_checkpoints:
            os.remove(path_network)
            os.remove(path_classifier)

    validation_metrics = compute_metrics(logits=jnp.mean(jnp.stack(preds), axis=0), labels=labels_total,
                                         num_classes=dataset_num_classes[cfg.hyperparameters.dataset_name])
    validation_metrics = jax.device_get(validation_metrics)

    #Std of predictions
    validation_metrics['std_logits'] = float(jnp.mean(jnp.std(jnp.stack(preds), axis=0)))

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
    paths_network = []
    paths_classifier = []
    for dir in subdirectories:
        if dir.isnumeric():
            paths_network.append(path_to_ensemble + '/' + dir + '/ckpts/network_0')
            paths_classifier.append(path_to_ensemble + '/' + dir + '/ckpts/classifier_0')

    validation_metrics = evaluate_saved_models(paths_network, paths_classifier, split_ds=split_ds, cfg=cfg)

    return validation_metrics


def eval_model(state_network, state_classifier, split_ds):
    """

    Parameters
    ----------
    state_network : train_state.TrainState
        The training state of the network.
    state_classifier : train_state.TrainState
        The training state of the classifier layer.
    split_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to some evaluation set.
    rng : jax.random.PRNGKey
        Pseudo-random number generator (PRNG) key for the random initialization of the neural network.

    Returns
    -------
    """

    logits = eval_step(state_network, state_classifier, split_ds)

    return logits, split_ds['label']


@jax.jit
def eval_step(state_network, state_classifier, batch):
    """
    A single evaluation step of the output logits of the neural network for a batch of inputs, as well as the
    cross-entropy loss and the classification accuracy.
    Parameters
    ----------
    state_network : train_state.TrainState
        The training state of the network.
    state_classifier : train_state.TrainState
        The training state of the classifier layer.
    batch : dict
        Dictionary with keys 'image' and 'label' corresponding to a batch of the training set.

    Returns
    -------
    metrics : dict
        A python dictionary with keys "loss" and "accuracy" corresponding to the cross-entropy loss and the accuracy
        for some logits and labels.
    """

    def per_sample_logits(params_network, params_classifier, x):
        out = state_network.apply_fn(params_network, x, train=False)
        out = state_classifier.apply_fn(params_classifier, out, train=False)
        return out
    return jax.vmap(per_sample_logits, (None, None, 0))(state_network.params, state_classifier.params, batch['image'])
