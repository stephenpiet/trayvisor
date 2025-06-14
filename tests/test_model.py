import unittest

import torch

from src.model import CNNVAE, MLPVAE
from tests import templates


class TestCNNVAE(unittest.TestCase, templates.ModelTestsMixin):
    def setUp(self):
        self.test_inputs = torch.randn(4, 1, 32, 32)
        self.net = CNNVAE(input_shape=(1, 32, 32), bottleneck_dim=16)

    def test_shape(self):
        outputs = self.net(self.test_inputs)
        # VAE returns (recon, mu, log_sigma)
        self.assertEqual(self.test_inputs.shape, outputs[0].shape)

    def test_batch_independence(self):
        inputs = self.test_inputs.clone()
        inputs.requires_grad = True

        # Compute forward pass in eval mode to deactivate batch norm
        self.net.eval()
        outputs = self.net(inputs)
        self.net.train()

        # Mask loss for certain samples in batch
        batch_size = inputs[0].shape[0]
        mask_idx = torch.randint(0, batch_size, ())
        mask = torch.ones_like(outputs[0])  # Use reconstruction output
        mask[mask_idx] = 0
        outputs = (outputs[0] * mask, outputs[1], outputs[2])

        # Compute backward pass
        loss = outputs[0].mean()  # Use reconstruction output
        loss.backward()

        # Check if gradient exists and is zero for masked samples
        for i, grad in enumerate(inputs.grad):
            if i == mask_idx:
                self.assertTrue(torch.all(grad == 0).item())
            else:
                self.assertTrue(not torch.all(grad == 0))

    def test_all_parameters_updated(self):
        optim = torch.optim.SGD(self.net.parameters(), lr=0.1)

        outputs = self.net(self.test_inputs)
        loss = outputs[0].mean()  # Use reconstruction output
        loss.backward()
        optim.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0.0, torch.sum(param.grad**2))


class TestMLPVAE(unittest.TestCase, templates.ModelTestsMixin):
    def setUp(self):
        self.test_inputs = torch.randn(4, 1, 32, 32)
        self.net = MLPVAE(input_shape=(1, 32, 32), bottleneck_dim=16)

    def test_shape(self):
        outputs = self.net(self.test_inputs)
        # VAE returns (recon, mu, log_sigma)
        self.assertEqual(self.test_inputs.shape, outputs[0].shape)

    def test_batch_independence(self):
        inputs = self.test_inputs.clone()
        inputs.requires_grad = True

        # Compute forward pass in eval mode to deactivate batch norm
        self.net.eval()
        outputs = self.net(inputs)
        self.net.train()

        # Mask loss for certain samples in batch
        batch_size = inputs[0].shape[0]
        mask_idx = torch.randint(0, batch_size, ())
        mask = torch.ones_like(outputs[0])  # Use reconstruction output
        mask[mask_idx] = 0
        outputs = (outputs[0] * mask, outputs[1], outputs[2])

        # Compute backward pass
        loss = outputs[0].mean()  # Use reconstruction output
        loss.backward()

        # Check if gradient exists and is zero for masked samples
        for i, grad in enumerate(inputs.grad):
            if i == mask_idx:
                self.assertTrue(torch.all(grad == 0).item())
            else:
                self.assertTrue(not torch.all(grad == 0))

    def test_all_parameters_updated(self):
        optim = torch.optim.SGD(self.net.parameters(), lr=0.1)

        outputs = self.net(self.test_inputs)
        loss = outputs[0].mean()  # Use reconstruction output
        loss.backward()
        optim.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0.0, torch.sum(param.grad**2))
