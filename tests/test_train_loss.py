"""Numerical regression test for the memory-efficient training loss."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from train import ChunkedLinearCrossEntropy


class ChunkedLossTests(unittest.TestCase):
    def test_loss_and_hidden_gradient_match_standard_cross_entropy(self) -> None:
        torch.manual_seed(7)
        hidden = torch.randn(2, 9, 6, requires_grad=True)
        weight = torch.randn(17, 6)
        labels = torch.tensor([
            [1, 2, -100, 4, 5, 6, 7, 8, 9],
            [2, -100, 3, 4, 5, 6, 1, 8, 10],
        ])

        expected = F.cross_entropy(
            F.linear(hidden, weight).reshape(-1, weight.shape[0]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        expected.backward()
        expected_gradient = hidden.grad.detach().clone()
        hidden.grad = None

        actual = ChunkedLinearCrossEntropy.apply(hidden, weight, labels, 4)
        actual.backward()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(hidden.grad, expected_gradient, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
