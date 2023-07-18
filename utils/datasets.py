import warnings
import tensorflow_datasets as tfds
import tensorflow as tf
from omegaconf import DictConfig
import jax.numpy as jnp
import jax
from jax import random


ood_dominoes = {
    'fashion_mnist': {'unlabeled': 10000, 'train': 45000, 'validation': 5000, 'test': 10000},
    'Cifar10': {'unlabeled': 10000, 'train': 35000, 'validation': 5000, 'test': 10000},
    'mnist': {'unlabeled': 10000, 'train': 45000, 'validation': 5000, 'test': 10000}
}

labels_dominoes = {
    'fashion_mnist': {'ood': [7, 8], 'id': [3, 4]},
    'Cifar10': {'ood': [3, 4], 'id': [1, 9]},
    'mnist': {'ood': [0, 1], 'id': [0, 1]}
}

dataset_dimensions = {
    'Cifar10': [32, 32, 3],
    'Cifar100': [32, 32, 3],
    'svhn_cropped': [32, 32, 3],
    'fashion_mnist': [28, 28, 1],
    'mnist': [28, 28, 1],
    'Cifar10_dominoes': [64, 32, 3],
    'fashion_mnist_dominoes': [56, 28, 1]
}

dataset_num_classes = {
    'Cifar10': 10,
    'Cifar100': 100,
    'svhn_cropped': 10,
    'fashion_mnist': 10,
    'Cifar10_dominoes': 10,
    'fashion_mnist_dominoes': 10
}


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
    unlabeled_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the unlabeled set.
    validation_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the validation set.
    """
    if cfg.hyperparameters.dataset_name not in ['Cifar10', 'Cifar100', 'svhn_cropped', 'fashion_mnist']:
        warnings.warn(cfg.hyperparameters.dataset_name+' might not exist in tensorflow_datasets. These experiments have been created for datasets ``Cifar10``, ``Cifar100``, ``svhn_cropped`` and ``fashion_mnist``.')

    if cfg.server.dataset_dir == 'default':
        ds_builder = tfds.builder(cfg.hyperparameters.dataset_name)
        ds_builder.download_and_prepare()
        train_ds_tmp = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
        test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    else:
        train_ds_tmp = tfds.load(name=cfg.hyperparameters.dataset_name, data_dir=cfg.server.dataset_dir, split='train',
                             batch_size=-1)
        test_ds = tfds.load(name=cfg.hyperparameters.dataset_name, data_dir=cfg.server.dataset_dir, split='test',
                            batch_size=-1)

    train_ds_tmp['image'] = jnp.float32(train_ds_tmp['image'])/255.
    test_ds['image'] = jnp.float32(test_ds['image'])/255.

    train_ds_tmp['label'] = jnp.int32(train_ds_tmp['label'])
    test_ds['label'] = jnp.int32(test_ds['label'])

    train_ds_tmp = {i: train_ds_tmp[i] for i in train_ds_tmp if i != 'id' and i != 'coarse_label'}
    test_ds = {i: test_ds[i] for i in test_ds if i != 'id' and i != 'coarse_label'}

    validation_ds = {}
    unlabeled_ds = {}
    train_ds = {}

    validation_ds['image'] = train_ds_tmp['image'][0:cfg.hyperparameters.size_validation]
    validation_ds['label'] = train_ds_tmp['label'][0:cfg.hyperparameters.size_validation]

    train_ds['image'] = train_ds_tmp['image'][cfg.hyperparameters.size_validation:cfg.hyperparameters.size_validation
                                                                              +cfg.hyperparameters.size_training]
    train_ds['label'] = train_ds_tmp['label'][cfg.hyperparameters.size_validation:cfg.hyperparameters.size_validation
                                                                              +cfg.hyperparameters.size_training]

    if cfg.hyperparameters.in_distribution:

        unlabeled_ds['image'] = train_ds_tmp['image'][
                                cfg.hyperparameters.size_validation + cfg.hyperparameters.size_training:]
        unlabeled_ds['label'] = train_ds_tmp['label'][
                                cfg.hyperparameters.size_validation + cfg.hyperparameters.size_training:]

    else:


        if cfg.server.dataset_dir == 'default':
            ds_builder = tfds.builder(cfg.hyperparameters.unlabeled_dataset_name)
            ds_builder.download_and_prepare()
            unlabeled_ds = tfds.as_numpy(ds_builder.as_dataset(split='unlabelled', batch_size=-1))
        else:
            unlabeled_ds = tfds.load(name=cfg.hyperparameters.unlabeled_dataset_name,
                                 data_dir=cfg.server.dataset_dir,
                                 split='unlabelled',
                                 batch_size=-1)

        # Resize to match the target distribution
        unlabeled_ds['image'] = tf.image.resize(unlabeled_ds['image'], [dataset_dimensions[cfg.hyperparameters.dataset_name][0],
                                        dataset_dimensions[cfg.hyperparameters.dataset_name][0]])
        # Turn to grayscale if the target distribution is greyscale
        if dataset_dimensions[cfg.hyperparameters.dataset_name][2] == 1:
            unlabeled_ds['image'] = tf.image.rgb_to_grayscale(
                unlabeled_ds['image'], name=None
            )
        unlabeled_ds['image'] = jnp.float32(unlabeled_ds['image']) / 255.
        unlabeled_ds['label'] = jnp.int32(unlabeled_ds['label'])
        unlabeled_ds = {i: unlabeled_ds[i] for i in unlabeled_ds if i != 'id'}


    if cfg.hyperparameters.mode == 'diverse':
        if cfg.hyperparameters.size_unlabeled > 0:
            unlabeled_ds['image'] = unlabeled_ds['image'][:cfg.hyperparameters.size_unlabeled]
            unlabeled_ds['label'] = unlabeled_ds['label'][:cfg.hyperparameters.size_unlabeled]

    return train_ds, test_ds, unlabeled_ds, validation_ds


def combine_dataset(ds_mnist, ds_other, cfg, in_distribution, randomized):
    '''
    Combines the mnist dataset with either the fashion_mnist or the Cifar10 dataset. If the in_distribution variable
    is True then we generate in distribution samples, otherwise we generate out-of-distribution samples. If the randomized
    variable is True then the digits from mnist are shuffled with the images from fashion_mnist or Cifar10. Otherwise,
    the digits 0 and 1 correlate with the cars and trucks classes from Cifar10 or the coats and dresses classes from
    fashion_mnist.

    Parameters
    ----------
    ds_mnist
    ds_other
    cfg
    in_distribution
    randomized

    Returns
    -------

    '''
    if in_distribution:
        in_distribution = 'id'
    else:
        in_distribution = 'ood'

    if cfg.hyperparameters.dataset_name == 'fashion_mnist_dominoes':
        other = 'fashion_mnist'
    elif cfg.hyperparameters.dataset_name == 'Cifar10_dominoes':
        other = 'Cifar10'

    ds_mnist['image'] = ds_mnist['image'][jnp.where((ds_mnist['label'] == labels_dominoes['mnist']['id'][0]) |
                                                    (ds_mnist['label'] == labels_dominoes['mnist']['id'][1]))[0]]
    ds_mnist['label'] = ds_mnist['label'][jnp.where((ds_mnist['label'] == labels_dominoes['mnist']['id'][0]) |
                                                    (ds_mnist['label'] == labels_dominoes['mnist']['id'][1]))[0]]

    ds_other['image'] = ds_other['image'][jnp.where((ds_other['label'] == labels_dominoes[other][in_distribution][0]) |
                                                    (ds_other['label'] == labels_dominoes[other][in_distribution][1]))[0]]
    ds_other['label'] = ds_other['label'][jnp.where((ds_other['label'] == labels_dominoes[other][in_distribution][0]) |
                                                    (ds_other['label'] == labels_dominoes[other][in_distribution][1]))[0]]

    if randomized:
        combined = {}
        min_dims = min(ds_mnist['label'].shape[0], ds_other['label'].shape[0])
        combined['image'] = jnp.concatenate([ds_mnist['image'][0: min_dims], ds_other['image'][0: min_dims]], axis=1)
        combined['label'] = ds_other['label'][0: min_dims]
    else:
        blocks = []
        for label_mnist, label_other in zip(labels_dominoes['mnist']['id'], labels_dominoes[other][in_distribution]):
            tmp = {}
            ds_mnist_tmp = {}
            ds_other_tmp = {}

            ds_mnist_tmp['image'] = ds_mnist['image'][ds_mnist['label'] == label_mnist]
            ds_mnist_tmp['label'] = ds_mnist['label'][ds_mnist['label'] == label_mnist]

            ds_other_tmp['image'] = ds_other['image'][ds_other['label'] == label_other]
            ds_other_tmp['label'] = ds_other['label'][ds_other['label'] == label_other]

            min_dims = min(ds_mnist_tmp['label'].shape[0], ds_other_tmp['label'].shape[0])
            tmp['image'] = jnp.concatenate([ds_mnist_tmp['image'][0: min_dims], ds_other_tmp['image'][0: min_dims]], axis=1)
            tmp['label'] = ds_other_tmp['label'][0: min_dims]
            blocks.append(tmp)

        combined = {}
        combined['image'] = jnp.concatenate([blocks[0]['image'], blocks[1]['image']], axis=0)
        combined['label'] = jnp.concatenate([blocks[0]['label'], blocks[1]['label']], axis=0)

    return combined


def combine_all_datasets(ds_mnist, ds_other, cfg):
    '''
    Takes as input the `mnist` dataset and either the `fashion_mnist` or the `Cifar10` dataset and creates the
    corresponding dominoes component

    Parameters
    ----------
    ds_mnist
    ds_other
    cfg
    randomized

    Returns
    -------

    '''
    combined_ds = {}
    for mnist_key, other_key in zip(ds_mnist.keys(), ds_other.keys()):
        if mnist_key == 'train':
            combined_ds[mnist_key] = combine_dataset(ds_mnist[mnist_key],
                                                     ds_other[mnist_key],
                                                     cfg,
                                                     in_distribution=True,
                                                     randomized=False,
                                                     )
        elif mnist_key == 'unlabeled':
            combined_ds[mnist_key] = combine_dataset(ds_mnist[mnist_key],
                                                     ds_other[mnist_key],
                                                     cfg,
                                                     in_distribution=cfg.hyperparameters.in_distribution,
                                                     randomized=True)
        elif mnist_key == 'test' or mnist_key == 'validation':
            combined_ds[mnist_key] = combine_dataset(ds_mnist[mnist_key],
                                                     ds_other[mnist_key],
                                                     cfg,
                                                     in_distribution=True,
                                                     randomized=True)
    return combined_ds


def preprocess_mnist(set_ds):
    '''
    Preprocess mnist to match the dimensions and number of channels of cifar10.
    Returns
    -------
    '''

    # Resize to match the target distribution
    set_ds['image'] = tf.image.resize(set_ds['image'], [dataset_dimensions['Cifar10'][0],
                                                        dataset_dimensions['Cifar10'][0]
                                                        ])
    # Turn to grayscale if the target distribution is greyscale
    set_ds['image'] = tf.image.grayscale_to_rgb(
        set_ds['image'], name=None
    )
    return set_ds


def load_single_dataset(cfg, dataset_name):
    '''

    Load a single dataset. This will be used together with other datasets to construct an ood task.
    Parameters
    ----------
    cfg
    dataset_name

    Returns
    -------

    '''
    if cfg.server.dataset_dir == 'default':
        ds_builder = tfds.builder(dataset_name)
        ds_builder.download_and_prepare()
        train_ds_tmp = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
        test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    else:
        train_ds_tmp = tfds.load(name=dataset_name, data_dir=cfg.server.dataset_dir, split='train',
                             batch_size=-1)
        test_ds = tfds.load(name=dataset_name, data_dir=cfg.server.dataset_dir, split='test',
                            batch_size=-1)


    if dataset_name == 'mnist' and cfg.hyperparameters.dataset_name == 'Cifar10_dominoes':
        train_ds_tmp = preprocess_mnist(train_ds_tmp)
        test_ds = preprocess_mnist(test_ds)



    train_ds_tmp['image'] = jnp.float32(train_ds_tmp['image'])/255.
    test_ds['image'] = jnp.float32(test_ds['image'])/255.

    train_ds_tmp['label'] = jnp.int32(train_ds_tmp['label'])
    test_ds['label'] = jnp.int32(test_ds['label'])

    train_ds_tmp = {i: train_ds_tmp[i] for i in train_ds_tmp if i != 'id'}
    test_ds = {i: test_ds[i] for i in test_ds if i != 'id'}

    validation_ds = {}
    train_ds = {}
    unlabeled_ds = {}

    validation_ds['image'] = train_ds_tmp['image'][0:ood_dominoes[dataset_name]['validation']]
    validation_ds['label'] = train_ds_tmp['label'][0:ood_dominoes[dataset_name]['validation']]
    train_ds['image'] = train_ds_tmp['image'][ood_dominoes[dataset_name]['validation']:
                                              ood_dominoes[dataset_name]['validation']
                                              + ood_dominoes[dataset_name]['train']]
    train_ds['label'] = train_ds_tmp['label'][ood_dominoes[dataset_name]['validation']:
                                              ood_dominoes[dataset_name]['validation']
                                              + ood_dominoes[dataset_name]['train']]
    unlabeled_ds['image'] = train_ds_tmp['image'][ood_dominoes[dataset_name]['validation']
                                                  + ood_dominoes[dataset_name]['train']:
                                                  ood_dominoes[dataset_name]['validation']
                                                  + ood_dominoes[dataset_name]['train']
                                                  + ood_dominoes[dataset_name]['unlabeled']]
    unlabeled_ds['label'] = train_ds_tmp['label'][ood_dominoes[dataset_name]['validation']
                                                  + ood_dominoes[dataset_name]['train']:
                                                  ood_dominoes[dataset_name]['validation']
                                                  + ood_dominoes[dataset_name]['train']
                                                  + ood_dominoes[dataset_name]['unlabeled']]

    return train_ds, test_ds, unlabeled_ds, validation_ds


def get_simple_ood_datasets(cfg):
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
    unlabeled_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the unlabeled set.
    validation_ds: dict
        Dictionary with keys 'image' and 'label' corresponding to the validation set.
    """
    if cfg.hyperparameters.dataset_name not in ['fashion_mnist_dominoes', 'Cifar10_dominoes']:
        warnings.warn('These experiments have been created for datasets ``fashion_mnist_dominoes``and '
                      '``Cifar10_dominoes``.')

    train_ds_mnist, test_ds_mnist, unlabeled_ds_mnist, validation_ds_mnist = load_single_dataset(cfg, 'mnist')

    if cfg.hyperparameters.dataset_name == 'fashion_mnist_dominoes':

        train_ds_other, test_ds_other, unlabeled_ds_other, validation_ds_other = load_single_dataset(cfg, 'fashion_mnist')

    elif cfg.hyperparameters.dataset_name == 'Cifar10_dominoes':

        train_ds_other, test_ds_other, unlabeled_ds_other, validation_ds_other = load_single_dataset(cfg, 'Cifar10')


    ds_mnist = {
        'train': train_ds_mnist,
        'test': test_ds_mnist,
        'unlabeled': unlabeled_ds_mnist,
        'validation': validation_ds_mnist
    }

    ds_other = {
        'train': train_ds_other,
        'test': test_ds_other,
        'unlabeled': unlabeled_ds_other,
        'validation': validation_ds_other
    }

    ds = combine_all_datasets(ds_mnist, ds_other, cfg)

    return ds['train'], ds['test'], ds['unlabeled'], ds['validation']


def flip_left_right(img):
    """Flips an image left/right direction."""
    return jnp.fliplr(img)


def identity(img):
    """Returns an image as it is."""
    return img


def random_horizontal_flip(img, flip):
    """Randomly flip an image vertically.

    Args:
        img: Array representing the image
        flip: Boolean for flipping or not
    Returns:
        Flipped or an identity image
    """

    return jax.lax.cond(flip, flip_left_right, identity, img)


def pad(img):
    """Pads images with 4 pixels on all dimensions."""
    return jnp.pad(img, ((0, 0), (4, 4), (4, 4), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))


def crop(img, pos1, pos2):
    """Crops an image to size 32 x 32"""
    return jax.lax.dynamic_slice(img, (pos1, pos2, 0), (32, 32, 3))


random_horizontal_flip_vmapped = jax.jit(jax.vmap(random_horizontal_flip, in_axes=(0, 0)))
crop_vmapped = jax.jit(jax.vmap(crop, in_axes=(0, 0, 0)))


def full_random_flip_function(batch, key):
    """
    The complete random flit function. Applies random flips to a minibatch.
    Parameters
    ----------
    batch: jnp.array
        An array of shape [batch_num, dim_1, dim_2, channel_num] which is a batch of training data.
    key: jax.random.PRNGKey
        A jax random PRNG key

    Returns
    -------
        : jnp.array
        An array of shape [batch_num, dim_1, dim_2, channel_num] which is a batch of training data flipped randomly.
    """
    flip = random.randint(key, shape=[batch.shape[0]], minval=0, maxval=2)
    return random_horizontal_flip_vmapped(batch, flip)


def full_random_crop_function(batch, key):
    """
    The complete random flit function. Applies random flips to a minibatch.
    Parameters
    ----------
    batch: jnp.array
        An array of shape [batch_num, dim_1, dim_2, channel_num] which is a batch of training data.
    key: jax.random.PRNGKey
        A jax random PRNG key

    Returns
    -------
        : jnp.array
        An array of shape [batch_num, dim_1, dim_2, channel_num] which is a batch of training data cropped randomly.
    """
    batch = pad(batch)
    key, subkey = random.split(key)
    pos1 = random.randint(subkey, shape=[batch.shape[0]], minval=0, maxval=8)
    key, subkey = random.split(key)
    pos2 = random.randint(subkey, shape=[batch.shape[0]], minval=0, maxval=8)
    return crop_vmapped(batch, pos1, pos2)