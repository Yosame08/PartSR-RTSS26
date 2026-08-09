import torch
from torch import nn
import torch.nn.functional as F


TEXTURE_CONTEXT_SIZE = 9
ARM_CONTEXT_SIZE = 3


def scheduler_context_size(action_count: int) -> int:
    if action_count <= 0:
        raise ValueError(f"action_count must be positive, got {action_count}")
    return TEXTURE_CONTEXT_SIZE + ARM_CONTEXT_SIZE + action_count


class Model(nn.Module):
    def __init__(self, k, input_size, hidden_size, out_size):
        super().__init__()
        assert input_size % 2 == 0
        self.texture_size = TEXTURE_CONTEXT_SIZE
        self.arm_features = ARM_CONTEXT_SIZE
        self.context_size = scheduler_context_size(k)
        self.texture_encoder = nn.Sequential(
            nn.Linear(self.texture_size, input_size * 2),
            nn.LeakyReLU(),
            nn.Linear(input_size * 2, input_size),
            nn.LeakyReLU(),
        )

        self.arm_encoder = nn.Sequential(
            nn.Linear(self.arm_features, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
        )

        self.one_hot_encoder = nn.Sequential(
            nn.Linear(k, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
        )

        self.affine = nn.Sequential(
            nn.Linear(input_size * 3, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, input_contexts):
        if input_contexts.dim() == 1:
            input_contexts = input_contexts.unsqueeze(0)
        if input_contexts.shape[-1] != self.context_size:
            raise ValueError(
                f"Expected scheduler context size {self.context_size}, got {input_contexts.shape[-1]}"
            )

        texture_context = input_contexts[:, :self.texture_size]
        arm_features = input_contexts[:, self.texture_size:self.texture_size + self.arm_features]
        one_hot_context = input_contexts[:, self.texture_size + self.arm_features:]

        texture_tensor = (self.texture_encoder(texture_context))
        arm_tensors = (self.arm_encoder(arm_features))
        one_hot_tensors = (self.one_hot_encoder(one_hot_context))

        combined = torch.cat((texture_tensor, arm_tensors, one_hot_tensors), dim=1)
        return self.affine(combined)
