from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dreamgrid.model import LatentWorldModel, WorldModelConfig, require_torch


def train(
    dataset: Path,
    out: Path,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    seed: int = 0,
    device_name: str = "auto",
    sample_dir: Path | None = None,
) -> Path:
    torch = require_torch()
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    device = _resolve_device(torch, device_name)
    data = np.load(dataset)
    obs = torch.tensor(data["obs"]).float().permute(0, 3, 1, 2) / 255.0
    next_obs = torch.tensor(data["next_obs"]).float().permute(0, 3, 1, 2) / 255.0
    actions = torch.tensor(data["actions"]).long()
    rewards = torch.tensor(data["rewards"]).float()
    dones = torch.tensor(data["dones"]).float()

    total = len(actions)
    val_count = max(1, int(total * val_fraction))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator)
    val_idx = permutation[:val_count]
    train_idx = permutation[val_count:]

    train_ds = TensorDataset(
        obs[train_idx],
        actions[train_idx],
        next_obs[train_idx],
        rewards[train_idx],
        dones[train_idx],
    )
    val_ds = TensorDataset(
        obs[val_idx],
        actions[val_idx],
        next_obs[val_idx],
        rewards[val_idx],
        dones[val_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = LatentWorldModel(WorldModelConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []

    print(
        f"device={device} train_examples={len(train_ds)} val_examples={len(val_ds)} "
        f"batch_size={batch_size}"
    )

    for epoch in range(epochs):
        model.train()
        running = 0.0
        example_count = 0
        for batch_obs, batch_action, batch_next, batch_reward, batch_done in train_loader:
            batch_obs = batch_obs.to(device)
            batch_action = batch_action.to(device)
            batch_next = batch_next.to(device)
            batch_reward = batch_reward.to(device)
            batch_done = batch_done.to(device)
            pred = model(batch_obs, batch_action)
            recon = F.mse_loss(pred["next_obs"], batch_next)
            reward_loss = F.mse_loss(pred["reward"], batch_reward)
            done_loss = F.binary_cross_entropy_with_logits(pred["done_logit"], batch_done)
            loss = recon + 0.1 * reward_loss + 0.1 * done_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(batch_action)
            example_count += len(batch_action)

        train_loss = running / max(1, example_count)
        val_metrics = _evaluate(model, val_loader, device)
        epoch_metrics = {"epoch": epoch + 1, "train_loss": train_loss, **val_metrics}
        history.append(epoch_metrics)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['val_loss']:.5f} "
            f"frame_mse={val_metrics['frame_mse']:.5f} "
            f"reward_mae={val_metrics['reward_mae']:.5f} "
            f"done_acc={val_metrics['done_acc']:.3f}"
        )
        if sample_dir is not None:
            _export_samples(model, val_ds, sample_dir, epoch + 1, device)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.__dict__,
            "state_dict": model.cpu().state_dict(),
            "history": history,
            "dataset": str(dataset),
            "final_metrics": history[-1] if history else {},
        },
        out,
    )
    return out


def _resolve_device(torch_module, device_name: str):
    if device_name == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    if device_name == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device_name == "mps" and not torch_module.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
    return torch_module.device(device_name)


def _evaluate(model, loader, device) -> dict[str, float]:
    torch = require_torch()
    from torch.nn import functional as F

    model.eval()
    total_loss = 0.0
    frame_mse = 0.0
    reward_abs = 0.0
    done_correct = 0
    count = 0

    with torch.no_grad():
        for batch_obs, batch_action, batch_next, batch_reward, batch_done in loader:
            batch_obs = batch_obs.to(device)
            batch_action = batch_action.to(device)
            batch_next = batch_next.to(device)
            batch_reward = batch_reward.to(device)
            batch_done = batch_done.to(device)
            pred = model(batch_obs, batch_action)
            recon = F.mse_loss(pred["next_obs"], batch_next)
            reward_loss = F.mse_loss(pred["reward"], batch_reward)
            done_loss = F.binary_cross_entropy_with_logits(pred["done_logit"], batch_done)
            loss = recon + 0.1 * reward_loss + 0.1 * done_loss

            batch_count = len(batch_action)
            total_loss += float(loss.detach()) * batch_count
            frame_mse += float(recon.detach()) * batch_count
            reward_abs += float((pred["reward"] - batch_reward).abs().mean().detach()) * batch_count
            done_pred = torch.sigmoid(pred["done_logit"]) > 0.5
            done_correct += int((done_pred == batch_done.bool()).sum().detach().cpu())
            count += batch_count

    return {
        "val_loss": total_loss / max(1, count),
        "frame_mse": frame_mse / max(1, count),
        "reward_mae": reward_abs / max(1, count),
        "done_acc": done_correct / max(1, count),
    }


def _export_samples(model, dataset, sample_dir: Path, epoch: int, device, max_samples: int = 8) -> None:
    torch = require_torch()

    sample_dir.mkdir(parents=True, exist_ok=True)
    count = min(max_samples, len(dataset))
    obs, actions, next_obs, _, _ = zip(*(dataset[idx] for idx in range(count)), strict=False)
    batch_obs = torch.stack(list(obs)).to(device)
    batch_actions = torch.stack(list(actions)).to(device)
    batch_next = torch.stack(list(next_obs))

    model.eval()
    with torch.no_grad():
        pred_next = model(batch_obs, batch_actions)["next_obs"].cpu()

    tile_size = batch_next.shape[-1]
    sheet = Image.new("RGB", (tile_size * 4, tile_size * count), "white")
    for idx in range(count):
        current = _tensor_to_image(batch_obs[idx].cpu())
        target = _tensor_to_image(batch_next[idx])
        predicted = _tensor_to_image(pred_next[idx])
        error = _error_to_image(pred_next[idx], batch_next[idx])
        y = idx * tile_size
        sheet.paste(current, (0, y))
        sheet.paste(target, (tile_size, y))
        sheet.paste(predicted, (tile_size * 2, y))
        sheet.paste(error, (tile_size * 3, y))
    sheet.save(sample_dir / f"epoch_{epoch:03d}_samples.png")


def _tensor_to_image(tensor) -> Image.Image:
    array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _error_to_image(pred, target) -> Image.Image:
    error = (pred.clamp(0, 1) - target.clamp(0, 1)).abs().mean(dim=0).numpy()
    scaled = np.clip(error * 8.0, 0.0, 1.0)
    array = np.zeros((scaled.shape[0], scaled.shape[1], 3), dtype=np.uint8)
    array[..., 0] = (scaled * 255).astype(np.uint8)
    array[..., 1] = ((1.0 - scaled) * 80).astype(np.uint8)
    array[..., 2] = ((1.0 - scaled) * 120).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DreamGrid latent world model.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--sample-dir", type=Path, default=None)
    args = parser.parse_args()
    path = train(
        dataset=args.dataset,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device_name=args.device,
        sample_dir=args.sample_dir,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
