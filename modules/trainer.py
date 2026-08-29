"""Generic training loop, shared across all four model types. Each model
only needs to implement `training_step(batch) -> (loss, log_dict)` and
`parameters()`; everything else (optimizer, gradient clipping, epoch
logging, loss history) lives here."""
import time

import torch
from torch.utils.data import DataLoader

def train_model(model, dataset, method_name, dataset_name, cfg, logger):
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    log_every = max(1, cfg.epochs // 10)
    t_start = time.time()
    for epoch in range(cfg.epochs):
        sums, counts = {}, {}
        for batch in loader:
            opt.zero_grad()
            loss, logs = model.training_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1

        entry = {"epoch": epoch}
        for k in sorted(sums):
            entry[k] = sums[k] / max(1, counts[k])
        history.append(entry)
        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == cfg.epochs - 1:
            detail = " ".join(f"{k}={entry[k]:.5f}" for k in entry if k != "epoch")
            logger.info(f"[{dataset_name}/{method_name}] epoch {epoch + 1}/{cfg.epochs} {detail}")

    train_time = time.time() - t_start
    logger.info(f"[{dataset_name}/{method_name}] training finished in {train_time:.1f}s ({len(dataset)} examples, {cfg.epochs} epochs)")
    return history, train_time