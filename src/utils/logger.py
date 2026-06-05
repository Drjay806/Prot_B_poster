import os
import csv
from collections import defaultdict
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class TrainingLogger:
    """
    Logs training metrics to console, a CSV on Google Drive, and TensorBoard.

    Usage:
        logger = TrainingLogger(log_dir="/tmp/runs", csv_path=".../training_log.csv")
        logger.log({"loss/total": 0.42, "loss/mse": 0.18}, step=100, phase="pretrain")
        logger.check_health(step=100)
        logger.close()
    """

    def __init__(self, log_dir: str = "/tmp/runs", csv_path: Optional[str] = None):
        self.csv_path = csv_path
        self.history: Dict[str, list] = defaultdict(list)
        self.global_step = 0
        self._disc_zero_streak = 0
        self._gen_zero_streak = 0

        if _TB_AVAILABLE:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        if csv_path:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self._csv_file = open(csv_path, "a", newline="")
            self._csv_writer = None  # lazy init so we can write headers once

    def log(self, metrics: Dict[str, float], step: int, phase: str = ""):
        self.global_step = step
        for key, value in metrics.items():
            self.history[key].append((step, value))
            if self.writer:
                self.writer.add_scalar(key, value, step)

        if self.csv_path:
            row = {"step": step, "phase": phase, **metrics}
            if self._csv_writer is None:
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=list(row.keys()), extrasaction="ignore"
                )
                self._csv_file.seek(0, 2)  # seek to end
                if self._csv_file.tell() == 0:
                    self._csv_writer.writeheader()
            try:
                self._csv_writer.writerow(row)
                self._csv_file.flush()
            except ValueError:
                # fieldnames mismatch between phases — re-init writer
                self._csv_writer = None

    def log_pretrain_epoch(self, epoch: int, total: int, losses: Dict[str, float], val_cosine: Optional[float] = None):
        parts = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
        val_str = f" | val_cos={val_cosine:.4f}" if val_cosine is not None else ""
        print(f"[Pretrain Epoch {epoch}/{total}] {parts}{val_str}")

    def log_adv_epoch(self, epoch: int, total: int, metrics: Dict[str, float]):
        d_real = metrics.get("loss/disc_real", 0)
        d_fake = metrics.get("loss/disc_fake", 0)
        g = metrics.get("loss/gen", 0)
        dm = metrics.get("reward/distmult_mean", 0)
        ra = metrics.get("disc/real_acc", 0)
        fa = metrics.get("disc/fake_acc", 0)
        fmax = metrics.get("val/fmax_bp", None)
        fmax_str = f" | val_Fmax={fmax:.4f}" if fmax is not None else ""
        print(
            f"[Adv Epoch {epoch}/{total}] D_loss={d_real+d_fake:.3f} (real={d_real:.3f} fake={d_fake:.3f}) "
            f"| G_loss={g:.3f} | DistMult={dm:.3f} | D_acc=real:{ra*100:.0f}% fake:{fa*100:.0f}%{fmax_str}"
        )

    def log_rl_epoch(self, epoch: int, total: int, metrics: Dict[str, float], is_best: bool = False):
        r = metrics.get("reward/total_mean", 0)
        struct = metrics.get("reward/structural", 0)
        adv = metrics.get("reward/adversarial", 0)
        sem = metrics.get("reward/semantic", 0)
        hier = metrics.get("reward/hierarchy_penalty", 0)
        w3 = metrics.get("curriculum/w3", 0)
        gnorm = metrics.get("grad/gen_norm", 0)
        fmax = metrics.get("val/fmax_bp", None)
        fmax_str = f" | val_Fmax={fmax:.4f}" if fmax is not None else ""
        best_str = " *** NEW BEST" if is_best else ""
        print(
            f"[RL Epoch {epoch}/{total}] R={r:.3f} (struct={struct:.3f} adv={adv:.3f} sem={sem:.3f} "
            f"hier_pen={hier:.3f}) | w3={w3:.3f} | grad_norm={gnorm:.3f}{fmax_str}{best_str}"
        )

    def check_health(self, step: int):
        """Print warnings for common training failure modes."""
        warnings = []

        protein_norm = self._last("embed/protein_norm_mean")
        if protein_norm is not None and protein_norm < 0.01:
            warnings.append(f"  ⚠ EMBEDDING COLLAPSE: protein norm={protein_norm:.4f} < 0.01")

        gen_output_norm = self._last("embed/gen_output_norm")
        if gen_output_norm is not None and gen_output_norm < 0.01:
            warnings.append(f"  ⚠ GENERATOR COLLAPSE: output norm={gen_output_norm:.4f} < 0.01")

        disc_total = self._last("loss/disc_total")
        if disc_total is not None:
            if disc_total < 0.01:
                self._disc_zero_streak += 1
            else:
                self._disc_zero_streak = 0
            if self._disc_zero_streak >= 5:
                warnings.append(f"  ⚠ DISCRIMINATOR DOMINANCE: disc_loss≈0 for {self._disc_zero_streak*10} steps")

        gen_loss = self._last("loss/gen")
        if gen_loss is not None:
            if gen_loss < 0.01:
                self._gen_zero_streak += 1
            else:
                self._gen_zero_streak = 0
            if self._gen_zero_streak >= 5:
                warnings.append(f"  ⚠ MODE COLLAPSE: gen_loss≈0 for {self._gen_zero_streak*10} steps")

        gen_grad = self._last("grad/gen_norm")
        if gen_grad is not None and gen_grad > 5.0:
            warnings.append(f"  ⚠ GRADIENT EXPLOSION: gen_grad_norm={gen_grad:.2f} > 5.0 — consider lower lr")

        for w in warnings:
            print(f"[Health @ step {step}] {w}")

    def _last(self, key: str):
        vals = self.history.get(key)
        if vals:
            return vals[-1][1]
        return None

    def close(self):
        if self.writer:
            self.writer.close()
        if self.csv_path and hasattr(self, "_csv_file"):
            self._csv_file.close()
