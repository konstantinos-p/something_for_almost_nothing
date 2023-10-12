from unittest import TestCase
from metrics import get_confusion_matrix
from metrics import get_brier
from metrics import get_brier_decomposition
import jax.numpy as jnp
import numpy as np
import jax


def sample_pseudosoftmax_and_labels():

    list = []
    size_sample = 1000
    mistakes = 0

    sharpness = 10

    for i in range(10):
        list.append(np.random.uniform(0, 1, (size_sample, 1)))

    preds = np.concatenate(list, axis=1)
    preds = np.power(preds, sharpness)
    preds = preds/np.sum(preds, axis=1, keepdims=True)

    labels = np.argmax(preds, axis=1)
    labels[np.random.choice(labels.shape[0], mistakes)] = np.random.randint(0, 10, mistakes)

    return jnp.array(preds), jnp.array(labels)


class Test(TestCase):
    def test_confusion_matrix(self):

        true_labels = jnp.array([1, 2, 4])
        predicted_labels = jnp.array([2, 2, 4])

        out = get_confusion_matrix(true_labels, predicted_labels)

        self.fail()

    def test_brier(self):

        preds, labels = sample_pseudosoftmax_and_labels()

        brier = get_brier(preds=preds, targets=labels)

        self.fail()

    def test_brier_decomposition(self):

        preds, labels = sample_pseudosoftmax_and_labels()

        uncertainty, resolution, reliability = get_brier_decomposition(preds=preds, targets=labels)

        brier1 = reliability - resolution + uncertainty

        brier2 = get_brier(preds=preds, targets=labels)

        self.fail()