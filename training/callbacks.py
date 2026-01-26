import torch
from pytorch_lightning.callbacks import Callback


def _wandb_run(trainer):
    logger = trainer.logger
    if logger is None:
        return None
    return getattr(logger, "experiment", None)


class WandbImageSanityCallback(Callback):
    def __init__(self, n_samples: int = 4, frame_idxs=(0, 14, 29), tag: str = "debug/images"):
        self.n_samples = n_samples
        self.frame_idxs = list(frame_idxs)
        self.tag = tag
        self._logged_epoch = -1

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        if trainer.current_epoch == self._logged_epoch:
            return
        run = _wandb_run(trainer)
        if run is None or "images" not in batch:
            return

        import wandb
        from torchvision.utils import make_grid

        imgs = batch["images"].detach().cpu()  # [B,T,3,H,W] expected in [0,1]
        B, T, C, H, W = imgs.shape
        n = min(self.n_samples, B)
        idxs = [i for i in self.frame_idxs if 0 <= i < T]
        if not idxs:
            return

        tiles = []
        for b in range(n):
            for t in idxs:
                tiles.append(imgs[b, t].clamp(0, 1))

        grid = make_grid(tiles, nrow=len(idxs))
        run.log({self.tag: wandb.Image(grid, caption=f"epoch={trainer.current_epoch}")}, step=trainer.global_step)
        self._logged_epoch = trainer.current_epoch


class WandbGradNormCallback(Callback):
    def __init__(self, tag: str = "debug/grad_norm"):
        self.tag = tag

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not trainer.is_global_zero:
            return
        run = _wandb_run(trainer)
        if run is None:
            return

        total = 0.0
        for p in pl_module.parameters():
            if p.grad is None:
                continue
            total += p.grad.detach().float().pow(2).sum().item()
        run.log({self.tag: total ** 0.5}, step=trainer.global_step)