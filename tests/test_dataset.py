import unittest

from src.dataset import MyMNIST


class TestMyMNIST(unittest.TestCase):
    def setUp(self):
        self.dataset = MyMNIST()

    def test_train_data(self):
        self.assertGreater(len(self.dataset.train_data), 0)
        x, y = self.dataset.train_data[0]
        self.assertEqual(x.shape, (1, 32, 32))
        self.assertIsInstance(y, int)
