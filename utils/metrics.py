import jax.numpy as jnp
# From https://github.com/SamsungLabs/pytorch-ensembles/blob/master/metrics.py#L83
# The implementation of "Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning"
# arxiv: https://arxiv.org/abs/2002.06470


def get_ece(preds, targets, n_bins=15, **args):
    bin_boundaries = jnp.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = jnp.max(preds, 1), jnp.argmax(preds, 1)
    accuracies = (predictions == targets)

    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = jnp.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = jnp.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = jnp.mean(accuracies[in_bin])
            avg_confidence_in_bin = jnp.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += jnp.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)
    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers, avg_confs_in_bins
    return ece


def get_tace(preds, targets, n_bins=15, threshold=1e-3, **args):
    n_objects, n_classes = preds.shape

    res = 0.0
    for cur_class in range(n_classes):
        cur_class_conf = preds[:, cur_class]

        targets_sorted = targets[cur_class_conf.argsort()]
        cur_class_conf_sorted = jnp.sort(cur_class_conf)

        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]

        bin_size = len(cur_class_conf_sorted) // n_bins

        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins - 1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array
            bin_acc = (targets_sorted[bin_start_ind: bin_end_ind] == cur_class)
            bin_conf = cur_class_conf_sorted[bin_start_ind: bin_end_ind]
            avg_confidence_in_bin = jnp.mean(bin_conf)
            avg_accuracy_in_bin = jnp.mean(bin_acc)
            delta = jnp.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
            #             print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
            res += delta * bin_size / (n_objects * n_classes)

    return res


def get_brier(preds, targets, **args):
    one_hot_targets = jnp.zeros(preds.shape)
    one_hot_targets.at[jnp.arange(len(targets)), targets].set(1.0)
    return jnp.mean(jnp.sum((preds - one_hot_targets) ** 2, axis=1))
