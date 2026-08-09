import math
import tempfile
import unittest
from pathlib import Path

import torch

from scheduler.neuralucb import NeuralUCB


def make_raw_bandit(**overrides):
    options = {
        "d": 2,
        "K": 1,
        "hidden_size": 2,
        "raw": True,
        "lr": 1e-5,
        "lr_mode": "constant",
    }
    options.update(overrides)
    return NeuralUCB(**options)


class NeuralUCBInitializationTest(unittest.TestCase):
    def test_empty_model_path_intentionally_uses_random_initialization(self):
        bandit = make_raw_bandit(model_path="")
        self.assertEqual(bandit.K, 1)

    def test_explicit_missing_model_fails(self):
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            make_raw_bandit(model_path="/definitely/missing/neuralucb.pth")

    def test_incompatible_model_fails_instead_of_using_random_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "incompatible.pth"
            torch.save(make_raw_bandit(d=3).net.state_dict(), model_path)
            with self.assertRaises(RuntimeError):
                make_raw_bandit(model_path=str(model_path))

    def test_compatible_model_is_loaded_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "compatible.pth"
            expected = make_raw_bandit()
            torch.save(expected.net.state_dict(), model_path)
            actual = make_raw_bandit(model_path=str(model_path))
            for expected_param, actual_param in zip(expected.net.parameters(), actual.net.parameters()):
                self.assertTrue(torch.equal(expected_param, actual_param))

    def test_formal_lr_schedule_starts_at_configured_init_lr(self):
        bandit = make_raw_bandit(
            lr_mode="decay_then_oscillate",
            lr_scheduler_config={
                "init_lr": 5e-4,
                "decay_steps": 250,
                "cycle_period": 50,
                "amplitude": 0.9,
                "warmup_steps": 0,
            },
        )
        self.assertTrue(
            math.isclose(
                bandit.optimizer.param_groups[0]["lr"],
                5e-4,
                rel_tol=0,
                abs_tol=1e-12,
            )
        )

    def test_uses_open_source_sigma_initialization(self):
        bandit = make_raw_bandit(lamb=2.0)
        expected = torch.full((bandit.numel,), 2.0, device=bandit.sigma_inv.device)
        self.assertTrue(torch.equal(torch.diag(bandit.sigma_inv), expected))


if __name__ == "__main__":
    unittest.main()
