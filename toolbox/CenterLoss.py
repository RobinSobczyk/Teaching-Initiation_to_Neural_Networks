from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


def precision_mean(num_list: List[float]) -> Tuple[Union[int, float], int]:
    r"""Take a list of (numbers, weight), and compute recursively wieghted means of half lists"""
    n = len(num_list)
    if n > 1:
        v1, w1 = precision_mean(num_list[: n // 2])
        v2, w2 = precision_mean(num_list[n // 2 :])
        return ((w1 * v1 + w2 * v2) / (w1 + w2), w1 + w2)
    elif n == 1:
        return num_list[0]
    else:
        return (0, 0)


class CenterLoss(_Loss):
    r"""An implementation of Center Loss.
    Computes centers by classes, then the loss is the mean distance to that center per class.
    Centers are then updated at each epochs by taking the new center per class.
    """

    def __init__(
        self,
        num_class: int,
        output_filter: Optional[Callable[[Any], torch.Tensor]] = None,
        truth_filter: Optional[Callable[[Any], torch.Tensor]] = None,
        device: torch.device = "cpu",
    ):
        super(CenterLoss, self).__init__()
        self.num_class = num_class
        self.temp_centers = [[] for i in range(num_class)]
        self.center = [None for i in range(num_class)]
        self.output_filter = output_filter
        self.truth_filter = truth_filter
        self.device = device

    def change_device(self, device: torch.device) -> None:
        r"""Change device on which the parameters are stored"""
        self.device = device
        for c in self.center:
            try:
                c.to(self.device)
            except AttributeError:
                if c is None:
                    pass
                else:
                    raise
        for submean_list in self.temp_centers:
            for i in range(len(submean_list)):
                (subcenter, weight) = submean_list[i]
                submean_list[i] = (subcenter.to(self.device), weight)

    def initialize(self, model: torch.nn.Module, dataloader: DataLoader) -> None:
        r"""Compute the centers with a first pass in order to allow loss computation afterwards"""
        with torch.no_grad():
            for _, (batch_data, batch_truth) in enumerate(dataloader):
                if self.output_filter is None:
                    out = model(batch_data.to(self.device))
                else:
                    out = self.output_filter(model(batch_data.to(self.device)))
                if self.truth_filter is None:
                    classes = batch_truth.to(self.device)
                else:
                    classes = self.truth_filter(batch_truth).to(self.device)
                for i in range(self.num_class):
                    truth_idx = classes == i
                    n_i = torch.sum(truth_idx)
                    if n_i != 0:
                        self.temp_centers[i].append(
                            (torch.mean(out[truth_idx], 0), n_i)
                        )

    def epoch_update(self) -> None:
        r"""Replace the old centers by the mean of the new values.
        Computations of the new center is made with `precision_mean` in order to allow different batch size in a single dataset.
        """
        with torch.no_grad():
            for i in range(self.num_class):
                self.center[i] = precision_mean(self.temp_centers[i])[0]
                self.temp_centers[i] = []

    def forward(self, output: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        r"""Compute the loss for a batch while storing the per class barycenters of this batch"""
        loss_value = torch.Tensor([0]).to(self.device)
        for i in range(self.num_class):
            if output[truth == i].shape[0] != 0:
                loss_value += torch.sum(
                    torch.mean((output[truth == i] - self.center[i]) ** 2, 0)
                )
                self.temp_centers[i].append(
                    (torch.mean(output[truth == i].detach(), 0), torch.sum(truth == i))
                )
        return loss_value
