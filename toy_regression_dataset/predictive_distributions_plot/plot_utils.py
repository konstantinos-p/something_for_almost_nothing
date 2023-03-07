from matplotlib import pyplot as plt
from toy_experiment import create_dataset, evaluate_model_outputs, MLP
import os
from flax.training import train_state, checkpoints
import optax
from jax import random
import jax.numpy as jnp


def plot_predictives(path, axs, color=0, line_style='-'):
    x_training, y_training, y_test, x_test, x_unlabeled, y_unlabeled = create_dataset()
    x = jnp.linspace(-1, 11, 1000)
    x = jnp.expand_dims(x, axis=1)



    cmap = plt.colormaps["plasma"]
    axs.scatter(x_training, y_training, c='black')
    #axs.scatter(x_test, y_test, c='blue')

    mean = jnp.zeros(x_unlabeled.shape)

    subdirectories = os.listdir(path)
    preds = []
    for dir in subdirectories:

        #Create dummy state
        rng = random.PRNGKey(0)
        network = MLP()
        params = network.init(rng, jnp.ones([1, 1]), train=False)
        tx = optax.adamw(0.001)
        state = train_state.TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        #Restore state
        #print(path+dir+'ckpts/checkpoint_5000')
        restored_state = checkpoints.restore_checkpoint(ckpt_dir=path+dir+'/ckpts/checkpoint_5000',
                                                        target=state)

        #Compute output
        logit_outputs = evaluate_model_outputs(state=restored_state, input=x)
        preds.append(logit_outputs)
        #Plot

    concat_preds = jnp.concatenate(preds, axis=1)
    axs.plot(x, jnp.mean(concat_preds, axis=1), linewidth=5, c=cmap(color), linestyle=line_style)
    axs.fill_between(jnp.reshape(x, (-1)), jnp.mean(concat_preds, axis=1)+2*jnp.std(concat_preds, axis=1),
                     jnp.mean(concat_preds, axis=1)-2*jnp.std(concat_preds, axis=1),
                     color=cmap(color), alpha=0.2)
    return