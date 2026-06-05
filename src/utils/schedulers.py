import math


class CurriculumScheduler:
    """
    Linearly ramps a weight multiplier from 0.0 to 1.0 over `warmup_epochs`.
    Used to grow the semantic reward weight w3 during RL training so the model
    focuses on structural/adversarial plausibility before chasing exact semantic accuracy.
    """

    def __init__(self, warmup_epochs: int = 30):
        self.warmup_epochs = warmup_epochs

    def get_multiplier(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.warmup_epochs)


class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to `min_lr` over the rest.
    Used for CompGCN pre-training.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            factor = self.min_lr + 0.5 * (1.0 - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor
