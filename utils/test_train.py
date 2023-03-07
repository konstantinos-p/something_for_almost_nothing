from unittest import TestCase
from train import get_datasets
import hydra
from omegaconf import DictConfig
import os
from hydra import initialize, compose


class TestGetCountries(TestCase):
    def test_generate_datasets(self):
        """
        Executes the get_datasets function to see if the datasets can be loaded successfully.
        Returns
        -------
        """
        executed = True
        try:
            with initialize(version_base=None, config_path='conf'):
                cfg = compose(config_name="test_conf")
            train_ds, test_ds, unlabeled_ds, validation_ds = get_datasets(cfg)
        except BaseException as e:
            print(e)
            executed = False

        self.assertTrue(executed)
