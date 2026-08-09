# -------------------------------------------------------------------
# This file is adapted from the repository:
# https://github.com/wadx2019/Neural-Bandit
# Original author: wadx2019
# Original license: MIT License (as of July 15, 2025).
# -------------------------------------------------------------------
import logging
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from config import device_name
from .env_encoder import Model
from torch.optim.lr_scheduler import _LRScheduler

device = device_name
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecayThenOscillateLR(_LRScheduler):
    """
    学习率调度器：先指数衰减到基础值，然后在基础值附近周期性波动
    参数:
        optimizer: 优化器对象
        init_lr: 初始学习率
        base_lr: 基础学习率（衰减目标值）
        decay_steps: 衰减到基础学习率所需的步数
        cycle_period: 周期性波动的周期长度（步数）
        amplitude: 波动幅度（相对于基础学习率的比例）
        warmup_steps: 可选的热身步数
        last_epoch: 最后一个epoch索引
    """

    def __init__(self, optimizer, init_lr, base_lr, decay_steps,
                 cycle_period, amplitude=0.1, warmup_steps=0, last_epoch=-1):
        self.init_lr = init_lr
        self.base_lr = base_lr
        self.decay_steps = decay_steps
        self.cycle_period = cycle_period
        self.amplitude = amplitude
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch  # _LRScheduler initializes last_epoch at step 0

        # 热身阶段
        if step < self.warmup_steps:
            warmup_factor = step / self.warmup_steps
            return [self.init_lr * warmup_factor for _ in self.base_lrs]

        # 指数衰减阶段
        if step <= self.decay_steps:
            decay_factor = math.exp(-5 * (step - self.warmup_steps) /
                                    max(1, self.decay_steps - self.warmup_steps))
            current_lr = self.base_lr + (self.init_lr - self.base_lr) * decay_factor
            return [current_lr for _ in self.base_lrs]

        # 周期性波动阶段
        cycle_progress = (step - self.decay_steps) / self.cycle_period
        oscillation = self.amplitude * self.base_lr * math.cos(2 * math.pi * cycle_progress)
        return [self.base_lr + oscillation for _ in self.base_lrs]


class RawModel(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.leaky_relu(self.affine1(x))
        return self.affine2(x)


class ReplayBuffer:

    def __init__(self, d, capacity):
        self.buffer = {'context': np.zeros((capacity, d)), 'reward': np.zeros((capacity, 1))}
        self.capacity = capacity
        self.size = 0
        self.pointer = 0

    def add(self, context, reward):
        self.buffer['context'][self.pointer] = context
        self.buffer['reward'][self.pointer] = reward
        self.size = min(self.size + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, n):
        idx = np.random.randint(0, self.size, size=n)
        return self.buffer['context'][idx], self.buffer['reward'][idx]


class NeuralUCB:

    def __init__(self, d, K, model_path="", beta=0.5, lamb=1, encode_hidden=16, hidden_size=128, lr=5e-4,
                 reg=0.000125, raw=False, lr_mode="decay_then_oscillate", lr_scheduler_config=None):
        if lamb <= 0:
            raise ValueError(f"lamb must be positive, got {lamb}")
        if reg < 0:
            raise ValueError(f"reg must be non-negative, got {reg}")
        self.K = K
        self.T = 0
        self.reg = reg
        self.beta = beta
        self.lr_mode = lr_mode
        self.net = Model(K, encode_hidden, hidden_size, 1) if not raw else RawModel(d, hidden_size, 1)

        if model_path:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f'Scheduler model path "{model_path}" does not exist')
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            self.net.load_state_dict(state_dict)
            logger.info("Scheduler has been successfully initialized from %s", model_path)
        else:
            logger.info("Scheduler is using an intentionally random initialization")

        self.hidden_size = hidden_size
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.scheduler_params = {
            'init_lr': 0.1,  # 初始学习率（较大）
            'base_lr': lr,  # 基础学习率（原始lr值）
            'decay_steps': 250,  # 衰减到基础值所需的步数
            'cycle_period': 50,  # 周期性波动周期
            'amplitude': 0.9,  # 波动幅度（基础学习率的90%）
            'warmup_steps': 0,
        }
        if lr_scheduler_config is not None:
            unknown_params = set(lr_scheduler_config) - set(self.scheduler_params)
            if unknown_params:
                raise ValueError(f"Unknown LR scheduler parameters: {sorted(unknown_params)}")
            self.scheduler_params.update(lr_scheduler_config)
            self.scheduler_params['base_lr'] = lr

        if lr_mode == "decay_then_oscillate":
            self.scheduler = DecayThenOscillateLR(self.optimizer, **self.scheduler_params)
        elif lr_mode == "constant":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown lr_mode: {lr_mode}")

        self.numel = sum(w.numel() for w in self.net.parameters() if w.requires_grad)
        self.sigma_inv = lamb * torch.eye(self.numel, dtype=torch.float32, device=device)
        self.device = device
        self.net.to(device)

        self.theta0 = torch.cat(
            [w.flatten() for w in self.net.parameters() if w.requires_grad]
        ).detach().clone()
        self.replay_buffer = ReplayBuffer(d, 10000)

    def take_action(self, context):
        # if self.T < 50:
        #     return random.randint(0, self.K - 1)
        # context = torch.tensor(context, dtype=torch.float32)
        # context = context.to(self.device)
        g = np.zeros((self.K, self.numel), dtype=np.float32)

        for k in range(self.K):
            g[k] = self.grad(context[k]).cpu().numpy()

        # with torch.no_grad():
        #     p = self.net(context).cpu().numpy() + self.beta * np.sqrt(np.matmul(np.matmul(g[:, None, :], self.sigma_inv.cpu().numpy()), g[:, :, None])[:, 0, :])

        # action = np.argmax(p)
        with torch.no_grad():
            mean = self.net(context).squeeze()  # 保持在GPU上
            # 在GPU上进行矩阵运算
            g_tensor = torch.tensor(g, device=self.device)  # g已在GPU上，此步可省略
            variance = torch.einsum('ni,ij,nj->n', g_tensor, self.sigma_inv, g_tensor)
            p = mean + self.beta * torch.sqrt(variance)
        action = p.argmax().item()
        # if isinstance(self.net, Model):
        #     print("mean:", mean.cpu().numpy(), "std:", torch.sqrt(variance).cpu().numpy(), "p:", p.cpu().numpy())
        #     print(context[action])
        return action

    def grad(self, x):
        y = self.net(x)
        self.optimizer.zero_grad()
        y.backward()
        return torch.cat(
            [w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.net.parameters() if w.requires_grad]
        ).to(self.device)

    def update(self, context, action, reward):
        if isinstance(reward, list):
            for ac in range(len(reward)):
                self.replay_buffer.add(context[ac].cpu().numpy(), reward[ac])
                # print(context[ac], reward[ac])
        else:
            self.replay_buffer.add(context[action].cpu().numpy(), reward)
        self.sherman_morrison_update(self.grad(context[action, None]).unsqueeze(1))

        self.T += 1
        self.train()

    # def sherman_morrison_update(self, v):
    #     self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (1+v.T @ self.sigma_inv @ v)
    def sherman_morrison_update(self, v):
        numerator = self.sigma_inv @ v @ v.t() @ self.sigma_inv
        denominator = 1 + v.t() @ self.sigma_inv @ v
        self.sigma_inv -= numerator / denominator

    def train(self):
        if self.T > self.K * 5 and self.T % 1 == 0:
            for _ in range(2):
                x, y = self.replay_buffer.sample(128)
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
                y = torch.tensor(y, dtype=torch.float32).to(self.device).view(-1, 1)
                y_hat = self.net(x)
                data_loss = F.mse_loss(y_hat, y)
                anchor_loss = data_loss.new_zeros(())
                if self.reg > 0:
                    current_theta = torch.cat([
                        w.flatten() for w in self.net.parameters() if w.requires_grad
                    ])
                    anchor_loss = self.reg * torch.sum((current_theta - self.theta0) ** 2)
                loss = data_loss + anchor_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if isinstance(self.net, Model):
                print("Loss:", loss, "Anchor:", anchor_loss, "LR:", self.optimizer.param_groups[0]['lr'])

    def save_net(self, model_path):
        torch.save(self.net.state_dict(), model_path)
