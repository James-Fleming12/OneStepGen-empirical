import numpy as np
import torch
from torch.utils.data import Dataset

def sample_moons(n: int, noise: float = 0.05) -> np.ndarray:
    n1 = n // 2
    n2 = n - n1
    theta1 = np.random.uniform(0, np.pi, n1)
    x1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    theta2 = np.random.uniform(0, np.pi, n2)
    x2 = np.stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5], axis=1)
    x = np.concatenate([x1, x2], axis=0)
    x += np.random.normal(scale=noise, size=x.shape)
    np.random.shuffle(x)
    return x.astype(np.float32)

def sample_checkerboard(n: int) -> np.ndarray:
    samples = []
    total = 0
    while total < n:
        x = np.random.uniform(-4, 4, size=(n, 2))
        mask = (np.floor(x[:, 0]) + np.floor(x[:, 1])) % 2 == 0
        chosen = x[mask]
        samples.append(chosen)
        total += len(chosen)
    x = np.concatenate(samples, axis=0)[:n]
    return (x / 4.0).astype(np.float32)

def sample_8gaussians(n: int, scale: float = 2.0, std: float = 0.1) -> np.ndarray:
    centers = [(scale * np.cos(a), scale * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    idx = np.random.randint(0, 8, size=n)
    pts = np.array([centers[i] for i in idx], dtype=np.float32)
    pts += np.random.normal(scale=std, size=pts.shape).astype(np.float32)
    return (pts / scale).astype(np.float32)

def sample_25gaussians(n: int, scale: float = 2.0, std: float = 0.05) -> np.ndarray:
    grid = np.linspace(-scale, scale, 5)
    centers = [(x, y) for x in grid for y in grid]
    idx = np.random.randint(0, 25, size=n)
    pts = np.array([centers[i] for i in idx], dtype=np.float32)
    pts += np.random.normal(scale=std, size=pts.shape).astype(np.float32)
    return (pts / scale).astype(np.float32)

def sample_unequal_gmm(n: int, scale: float = 2.0) -> np.ndarray:
    # A GMM with 3 modes of unequal probabilities and variances, ideal for testing if minority modes are dropped
    centers = [(-scale, 0), (scale, 0), (0, scale)]
    stds = [0.1, 0.2, 0.05]
    probs = [0.7, 0.2, 0.1]
    
    idx = np.random.choice(3, size=n, p=probs)
    pts = np.array([centers[i] for i in idx], dtype=np.float32)
    noise = np.array([np.random.normal(scale=stds[i], size=2) for i in idx], dtype=np.float32)
    pts += noise
    return (pts / scale).astype(np.float32)

def sample_swissroll(n: int, noise: float = 0.05) -> np.ndarray:
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    x = t * np.cos(t)
    y = t * np.sin(t)
    pts = np.stack([x, y], axis=1) / 15.0
    pts += np.random.normal(scale=noise, size=pts.shape)
    return pts.astype(np.float32)

def sample_gmm(n: int, dim: int = 2, n_modes: int = 8, center_scale: float = 3.0,
               std_low: float = 0.05, std_high: float = 0.4) -> np.ndarray:
    # High-dimensional GMM: modes on a sphere of radius `center_scale`, each with a
    # different std. Varying scales in high dim stress distance-based drift methods.
    dirs = np.random.normal(size=(n_modes, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    centers = (center_scale * dirs).astype(np.float32)
    stds = np.linspace(std_low, std_high, n_modes)
    idx = np.random.choice(n_modes, size=n, p=np.ones(n_modes) / n_modes)
    pts = centers[idx]
    pts += np.random.normal(scale=stds[idx, None], size=(n, dim)).astype(np.float32)
    return pts.astype(np.float32)

SYNTHETIC_REGISTRY = {
    "moons": sample_moons,
    "checkerboard": sample_checkerboard,
    "8gaussians": sample_8gaussians,
    "25gaussians": sample_25gaussians,
    "unequal_gmm": sample_unequal_gmm,
    "swissroll": sample_swissroll,
    "gmm_16d": lambda n: sample_gmm(n, dim=16, n_modes=8, center_scale=3.0),
    "gmm_64d": lambda n: sample_gmm(n, dim=64, n_modes=8, center_scale=3.0),
}

# Dimensionality override for datasets whose samples are not 2D.
SYNTHETIC_DIMS = {"gmm_16d": 16, "gmm_64d": 64}

def sample_synthetic(name: str, n: int, seed: int = None) -> torch.Tensor:
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    x = SYNTHETIC_REGISTRY[name](n)
    if seed is not None:
        np.random.set_state(state)
    return torch.from_numpy(x)

class SyntheticDataset(Dataset):
    def __init__(self, name: str, n_samples: int = 20000, seed: int = 0):
        if name not in SYNTHETIC_REGISTRY:
            raise ValueError(f"Unknown synthetic dataset '{name}'. Choices: {sorted(SYNTHETIC_REGISTRY)}")
        self.name = name
        self.data = sample_synthetic(name, n_samples, seed=seed)
        self.dim = SYNTHETIC_DIMS.get(name, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]