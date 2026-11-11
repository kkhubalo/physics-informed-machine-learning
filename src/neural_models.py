"""Data-driven and physics-informed neural trajectory models."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

class TrajectoryNetwork(nn.Module):
    """Tanh MLP mapping physical time to displacement."""
    def __init__(self, start_time: float, end_time: float, width: int = 64):
        super().__init__()
        self.register_buffer("time_midpoint", torch.tensor((start_time + end_time) / 2))
        self.register_buffer("time_half_range", torch.tensor((end_time - start_time) / 2))
        self.layers = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width),
            nn.Tanh(), nn.Linear(width, 1))
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return self.layers((time - self.time_midpoint) / self.time_half_range)

@dataclass(frozen=True)
class NeuralFit:
    model: TrajectoryNetwork
    training_rmse: float
    validation_rmse: float
    damping: float | None = None
    stiffness: float | None = None

def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32).reshape(-1, 1)

def _split(length: int):
    train = torch.arange(length) % 5 != 0
    return train, ~train

def train_data_driven_model(times: np.ndarray, displacement: np.ndarray, *,
        epochs: int = 2000, learning_rate: float = 1e-3, seed: int = 42) -> NeuralFit:
    """Fit an MLP using observation loss only and a deterministic 80/20 split."""
    torch.manual_seed(seed)
    t, x = _tensor(times), _tensor(displacement)
    train, validation = _split(len(t))
    model = TrajectoryNetwork(float(times[0]), float(times[-1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(t[train]) - x[train]) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_rmse = torch.sqrt(torch.mean((model(t[train]) - x[train]) ** 2)).item()
        validation_rmse = torch.sqrt(torch.mean((model(t[validation]) - x[validation]) ** 2)).item()
    return NeuralFit(model, train_rmse, validation_rmse)

def train_physics_informed_model(times: np.ndarray, displacement: np.ndarray, *,
        mass: float = 1.0, initial_damping: float = 0.3,
        initial_stiffness: float = 3.0, physics_weight: float = 1.0,
        epochs: int = 2500, learning_rate: float = 1e-3, seed: int = 42) -> NeuralFit:
    """Train with data loss plus the damped-oscillator ODE residual."""
    if mass <= 0 or initial_damping <= 0 or initial_stiffness <= 0:
        raise ValueError("mass and initial parameter values must be positive")
    if physics_weight < 0:
        raise ValueError("physics_weight must be non-negative")
    torch.manual_seed(seed)
    t, x = _tensor(times), _tensor(displacement)
    train, validation = _split(len(t))
    model = TrajectoryNetwork(float(times[0]), float(times[-1]))
    log_damping = nn.Parameter(torch.tensor(float(np.log(initial_damping))))
    log_stiffness = nn.Parameter(torch.tensor(float(np.log(initial_stiffness))))
    optimizer = torch.optim.Adam(list(model.parameters()) + [log_damping, log_stiffness],
                                 lr=learning_rate)
    collocation = torch.linspace(float(times[0]), float(times[-1]), 256).reshape(-1, 1)
    residual_scale = max(initial_stiffness / mass, 1e-6)
    for _ in range(epochs):
        optimizer.zero_grad()
        data_loss = torch.mean((model(t[train]) - x[train]) ** 2)
        tp = collocation.detach().clone().requires_grad_(True)
        xp = model(tp)
        dx = torch.autograd.grad(xp, tp, torch.ones_like(xp), create_graph=True)[0]
        d2x = torch.autograd.grad(dx, tp, torch.ones_like(dx), create_graph=True)[0]
        damping, stiffness = torch.exp(log_damping), torch.exp(log_stiffness)
        residual = d2x + (damping / mass) * dx + (stiffness / mass) * xp
        physics_loss = torch.mean((residual / residual_scale) ** 2)
        (data_loss + physics_weight * physics_loss).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_rmse = torch.sqrt(torch.mean((model(t[train]) - x[train]) ** 2)).item()
        validation_rmse = torch.sqrt(torch.mean((model(t[validation]) - x[validation]) ** 2)).item()
        fitted_damping, fitted_stiffness = torch.exp(log_damping).item(), torch.exp(log_stiffness).item()
    return NeuralFit(model, train_rmse, validation_rmse, fitted_damping, fitted_stiffness)
