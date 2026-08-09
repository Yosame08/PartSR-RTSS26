from copy import deepcopy


FORMAL_SCHEDULER_EXPERIMENT = "A"
FORMAL_SCHEDULER_SEED = 42

_FORMAL_NEURAL_UCB_CONFIG = {
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
}


def formal_neural_ucb_config():
    return deepcopy(_FORMAL_NEURAL_UCB_CONFIG)
