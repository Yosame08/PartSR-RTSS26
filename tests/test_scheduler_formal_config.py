import unittest

from scheduler.formal_config import (
    FORMAL_SCHEDULER_EXPERIMENT,
    FORMAL_SCHEDULER_SEED,
    formal_neural_ucb_config,
)


class FormalSchedulerConfigTest(unittest.TestCase):
    def test_matches_server_a_configuration(self):
        self.assertEqual(FORMAL_SCHEDULER_EXPERIMENT, "A")
        self.assertEqual(FORMAL_SCHEDULER_SEED, 42)
        self.assertEqual(
            formal_neural_ucb_config(),
            {
                "beta": 0.5,
                "lamb": 1.0,
                "lr": 1e-5,
                "reg": 1.25e-4,
                "lr_mode": "decay_then_oscillate",
                "lr_scheduler_config": {
                    "init_lr": 5e-4,
                    "decay_steps": 250,
                    "cycle_period": 50,
                    "amplitude": 0.9,
                    "warmup_steps": 0,
                },
            },
        )

    def test_returns_an_independent_copy(self):
        first = formal_neural_ucb_config()
        first["lr_scheduler_config"]["init_lr"] = 1.0
        self.assertEqual(formal_neural_ucb_config()["lr_scheduler_config"]["init_lr"], 5e-4)


if __name__ == "__main__":
    unittest.main()
