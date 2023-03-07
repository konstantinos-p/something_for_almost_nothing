from unittest import TestCase
import models


class TestMLP(TestCase):
    def test_partial(self):
        fail = False
        try:
            model_cls = getattr(models, 'MLP_Small')
            full_model = model_cls(num_classes=10)
        except:
            fail=True
        self.assertFalse(fail)


