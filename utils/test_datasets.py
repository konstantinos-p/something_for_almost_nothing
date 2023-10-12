from unittest import TestCase
from datasets import get_cifar10_corrupted
from datasets import cifar10_corruptions
from datasets import cifar10_corruption_severities


class Test(TestCase):
    def test_get_cifar10_corrupted(self):

        for i in range(len(cifar10_corruptions)):
            for j in range(len(cifar10_corruption_severities)):
                test_ds = get_cifar10_corrupted(cifar10_corruptions[i], severity=cifar10_corruption_severities[j])
        self.fail()
