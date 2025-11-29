from typing import Mapping, Optional, Iterator, Any
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer(nn.Module):
    """ Trainer for classification models using PyTorch."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-4,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr)
        self.loss_fn = loss_fn
        self.device = device
        self.history: dict[str, list[float]] = {"epoch_loss": []}


    def train(self, dataloader: DataLoader, *, epochs: int = 100, silent: bool = False):
        self.history = {"epoch_loss": []}
        for _ in self.train_iter(dataloader, epochs=epochs, silent=silent):
            pass
        return self.history

    def train_iter(
        self,
        dataloader: DataLoader,
        *,
        epochs: int = 100,
        silent: bool = False,
    ) -> Iterator[nn.Module]:

        model = self.model.to(self.device)
        self._optimizer_to(self.optimizer, self.device)
        
        epoch_bar = tqdm(range(epochs), desc="Epochs", disable=silent)

                # epoch loop with outer tqdm
        for epoch in epoch_bar:
            model.train()
            running_loss = 0.0
            n_samples = 0

            # inner tqdm over batches so you see batch progress + loss
            batch_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
                disable=silent,
            )

            for data, target in batch_bar:
                x = data.to(self.device)
                y = target.to(self.device)

                self.optimizer.zero_grad()
                outputs = model(x)
                loss = self.loss_fn(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                bs = y.size(0)
                running_loss += loss.item() * bs
                n_samples += bs

                # show current loss in the batch progress bar
                batch_bar.set_postfix(loss=float(loss.detach().cpu()))
            
            # average loss for this epoch
            epoch_loss = running_loss / max(1, n_samples)
            self.history["epoch_loss"].append(epoch_loss)

            # show epoch loss in outer bar
            epoch_bar.set_postfix(loss=epoch_loss)
            
            # yield model after each epoch (unchanged behaviour)
            yield model 
            
    def _optimizer_to(self, optim_: torch.optim.Optimizer, device: torch.device) -> None:
        """Moves optimizer state tensors to the device."""
        for param in optim_.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        sd["optimizer"] = self.optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        del state_dict["optimizer"]
        super().load_state_dict(state_dict, strict, assign)