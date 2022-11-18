import torch
from torch.nn import functional as F
from edice.learners.base import BaseLearner


class BaseLearner:
    """Base learner class.

    A learner must implement a train_step method that performs
    parameter updates given a batch of data
    """
    def __init__(
        self,
        model,
        optimizer,
        device,
        lr_scheduler=None,
        max_grad_norm=5.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm

    def set_state(self, mode):
        if hasattr(self, "model"):
            if mode == "train":
                self.model.train()
            elif mode == "eval":
                self.model.eval()
            else:
                raise ValueError(f"mode must be train or eval but got {mode}")

    def epoch_begin(self):
        """Called at the start of each training epoch."""
        self.set_state("train")

    def train_begin(self):
        """Called at the start of a training loop."""
        # TODO add a reload option + method
        self._input_step = 0
        if not hasattr(self, "_update_step"):
            self._update_step = 0

    def test_begin(self):
        """Called at the start of a single pass through val/test data."""        
        self.set_state("eval")

    def __call__(self, batch):
        """Call forward_step on batch."""
        return self.forward_step(batch)

    def forward_step(self, batch, is_train=False):
        """Run a forward pass, computing loss and metrics on input batch.

        (loss is a tensor scalar and metrics a dict of python scalars)
        """
        raise NotImplementedError()

    def get_batch_size(self, batch):
        """Compute batch size (helps in metric accumulation)."""
        raise NotImplementedError() 

    def train_step(self, batch):
        loss, metrics = self.forward_step(batch, is_train=True)
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        self._update_step += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]

        self.optimizer.zero_grad()

        # TODO: add grad norm to metrics.

        return metrics

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def test_step(self, batch):
        """Compute metrics on a val/test batch."""
        loss, metrics = self.forward_step(batch, is_train=False)

        return metrics

    def checkpoint(self, output_dir, epoch, start_epoch=0):
        """Save model weights (and optimizer/scheduler state) to file.
        """
        file_ext = f".{'' if start_epoch == 0 else (str(start_epoch) + '.')}pt"
        os.makedirs(output_dir, exist_ok=True)  # needed b.c. can occur before save_config during first epoch.

        d = {
            "epoch": epoch,
            "weights": self.state_dict(),
            "step": self._update_step,
        }

        d["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            d["scheduler"] = self.lr_scheduler.state_dict()
        torch.save(d, os.path.join(output_dir, "checkpoint" + file_ext))


    def load_checkpoint_from_file(
        self,
        chkpt_file,
        device=None,
    ):
        # TODO handle device ...
        chkpt = torch.load(chkpt_file, map_location=device)
        if "step" in chkpt:
            self._update_step = chkpt["step"]
        self.load_state_dict(chkpt["weights"])
        self.optimizer.load_state_dict(chkpt["optimizer"])
        if self.lr_scheduler is None and "scheduler" in chkpt:
            print("Loaded scheduler state but scheduler must be passed to reinstantiate")
        elif "scheduler" in chkpt:
            print("Loaded scheduler")
            self.lr_scheduler.load_state_dict(chkpt["scheduler"])

    def load_checkpoint(self, output_dir, optimizer=None, scheduler=None, start_epoch=0):
        """Load model weights (and optimizer state) from file."""
        filename = f"checkpoint.{'' if start_epoch == 0 else (str(start_epoch) + '.')}pt"
        chkpt_file = os.path.join(output_dir, filename)
        return self.load_checkpoint_from_file(
            chkpt_file, device=self.device
        )            
