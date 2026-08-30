import torch

from ..networks import MLPVectorField, SmallUNet

class DiffusionModel:
    """Standard epsilon-prediction DDPM (linear noise schedule) trained on
    1,000 discrete timesteps and sampled with deterministic DDIM (eta=0),
    so the multi-step baseline uses the same 50-step budget as flow matching.
    The backbone is the same network family used by the other methods."""

    name = "diffusion"

    def __init__(self, data_info, device="cpu", hidden=256, depth=4, sample_steps=50,
                 timesteps=1000, beta_start=1e-4, beta_end=0.02, **kwargs):
        self.device = device
        self.data_info = data_info
        self.sample_steps = sample_steps
        self.timesteps = timesteps
        # x0 prediction is only clipped for images, where the data lives in [-1, 1].
        self.clamp_x0 = data_info["type"] == "image"

        if data_info["type"] == "synthetic":
            self.net = MLPVectorField(dim=data_info["dim"], hidden=hidden, depth=depth, use_r=False)
            self.shape = (data_info["dim"],)
        else:
            self.net = SmallUNet(channels=data_info["channels"], base=32, use_r=False)
            s = data_info["image_size"]
            self.shape = (data_info["channels"], s, s)
        self.net.to(device)

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def parameters(self):
        return self.net.parameters()

    def training_step(self, x1):
        x1 = x1.to(self.device)
        b = x1.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device)
        eps = torch.randn_like(x1)
        a_bar = self.alphas_cumprod[t].view(-1, *([1] * (x1.dim() - 1)))
        x_t = torch.sqrt(a_bar) * x1 + torch.sqrt(1 - a_bar) * eps
        pred = self.net(x_t, t.float() / self.timesteps)
        loss = ((pred - eps) ** 2).mean()
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def sample(self, n, steps=None):
        steps = min(steps or self.sample_steps, self.timesteps)
        x = torch.randn(n, *self.shape, device=self.device)
        # Deterministic DDIM (eta=0) over `steps` evenly spaced timesteps.
        t_seq = torch.linspace(self.timesteps - 1, 0, steps + 1).round().long().to(self.device)
        for i in range(steps):
            t = t_seq[i]
            t_next = t_seq[i + 1]
            t_batch = torch.full((n,), t.item(), device=self.device) / self.timesteps
            eps_pred = self.net(x, t_batch)
            a_t = self.alphas_cumprod[t]
            a_tn = self.alphas_cumprod[t_next]
            x0_pred = (x - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(a_t)
            if self.clamp_x0:
                x0_pred = x0_pred.clamp(-1, 1)
            x = torch.sqrt(a_tn) * x0_pred + torch.sqrt(1 - a_tn) * eps_pred
        return x

    def nfe_for_sampling(self, steps=None):
        return steps or self.sample_steps

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)
