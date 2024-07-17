from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.modules.loss import _Loss
from CenterLoss import CenterLoss

def auto_arg(x : Any, i : int) -> Any:
    r"""Get `i`-th item of `x` if `x` is a list or a tuple, and return `x` otherwise.
    """
    if isinstance(x, (list, tuple)):
        return x[i]
    else:
        return x

class MyCrossEntropyLoss(nn.CrossEntropyLoss):
    r"""Traditionnal Cross-entropy loss that supports datasets with hierarchical truth.
    """
    def init(
        self,
        weight : Optional[torch.Tensor]=None,
        size_average : Optional[Any]=None,
        ignore_index : int =-100,
        reduce : Optional[Any]=None,
        reduction : str='mean',
        label_smoothing : float=0.0
    ):
        super(MyCrossEntropyLoss).__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(input, target[:,0], weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

class MyHDL(_Loss):
    r"""Hierarchical loss relying on a CenterLoss for the first level of prediction and cross entropies for the other.
    Used for end-stacked prediction layers.
    Weights over hierarchy levels can be adjusted through `lambdas` parameter.
    """
    def __init__(
        self,
        hierarchy_size : Sequence[int],
        lambdas : Optional[Union[int, float, Sequence[Union[float, int]]]]=None,
        output_filter : Optional[Callable[[Any], torch.Tensor]]=None,
        truth_filter : Optional[Callable[[Any], torch.Tensor]]=None,
        device : torch.device ='cpu',
        weight : Optional[Union[torch.Tensor, Sequence[torch.Tensor]]]=None,
        size_average : Optional[Any]=None,
        ignore_index : Optional[Union[int, Sequence[int]]]=-100,
        reduce : Optional[Any]=None,
        reduction : Optional[Union[str, Sequence[str]]]='mean',
        label_smoothing : Optional[Union[float, Sequence[float]]]=0.0
    ):
        self.hierarchy_length=len(hierarchy_size)
        super(MyHDL, self).__init__()
        if lambdas is None:
            self.lambdas=1
        else:
            self.lambdas=lambdas
        self.device=device
        self.centerloss=CenterLoss(hierarchy_size[0], output_filter, truth_filter, device)
        self.crossentropies=nn.ModuleList([nn.CrossEntropyLoss(weight=auto_arg(weight,i),
                                                                size_average=auto_arg(size_average, i),
                                                                ignore_index=auto_arg(ignore_index, i),
                                                                reduce=auto_arg(reduce, i),
                                                                reduction=auto_arg(reduction, i),
                                                                label_smoothing=auto_arg(label_smoothing, i))
                                            for i in range(self.hierarchy_length-1)])

    def initialize(self, dataloader : DataLoader, model : torch.nn.Module) -> None:
        self.centerloss.initialize(dataloader, model)

    def epoch_update(self) -> None:
        self.centerloss.epoch_update()

    def forward(self, input : Sequence[torch.Tensor], target : torch.Tensor) -> torch.Tensor:
        loss_value=auto_arg(self.lambdas,0)*self.centerloss(input[0], target[:,0])
        for i in range(1,self.hierarchy_length):
            loss_value+=auto_arg(self.lambdas,i)*self.crossentropies[i-1](input[i], target[:,i])
        return loss_value

class MyScatteredHL(_Loss):
    r"""Cross-entropy loss for all the hierarchical predictions.
    Weights over hierarchy levels can be adjusted through `lambdas` parameter.
    """
    def __init__(
        self,
        hierarchy_size : Sequence[int],
        lambdas : Optional[Union[int, float, Sequence[Union[float, int]]]]=None,
        weight : Optional[Union[torch.Tensor, Sequence[torch.Tensor]]]=None,
        size_average : Optional[Any]=None,
        ignore_index : Optional[Union[int, Sequence[int]]]=-100,
        reduce : Optional[Any]=None,
        reduction : Optional[Union[str, Sequence[str]]]='mean',
        label_smoothing : Optional[Union[float, Sequence[float]]]=0.0
    ):
        super(MyScatteredHL, self).__init__()
        self.hierarchy_length=len(hierarchy_size)
        if lambdas is None:
            self.lambdas=1
        else:
            self.lambdas=lambdas
        self.crossentropies=nn.ModuleList([nn.CrossEntropyLoss(weight=auto_arg(weight,i),
                                                                size_average=auto_arg(size_average, i),
                                                                ignore_index=auto_arg(ignore_index, i),
                                                                reduce=auto_arg(reduce, i),
                                                                reduction=auto_arg(reduction, i),
                                                                label_smoothing=auto_arg(label_smoothing, i))
                                            for i in range(self.hierarchy_length)])

    def forward(self, input : Sequence[torch.Tensor], target : torch.Tensor) -> torch.Tensor:
        loss_value=auto_arg(self.lambdas,0)*self.crossentropies[0](input[0], target[:,0])
        for i in range(1,self.hierarchy_length):
            loss_value+=auto_arg(self.lambdas,i)*self.crossentropies[i](input[i], target[:,i])
        return loss_value

class AccuracyMeter(object):
    r"""A class meant to count the accuracy at different levels of hierarchy.
    """
    def __init__(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError
    
    def zero(self) -> Any:
        raise NotImplementedError

class ShallowAccuracyMeter(AccuracyMeter):
    r"""Accuracy metter counting only accuracy for the leaf level of hierarchy
    """
    def __init__(self, topk : int=1):
        super(ShallowAccuracyMeter, self).__init__()
        self.topk=topk

    def __call__(self, input : Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor,...]], target : torch.Tensor) -> int:
        results=input[0] if isinstance(input, (list,tuple)) else input
        temp_topk=results.topk(self.topk,1)[1].squeeze(-1)
        if temp_topk.shape!=target[:,0].shape:
            temp_target=target[:,0,None]
        else:
            temp_target=target[:,0]
        return torch.sum(temp_topk==temp_target).cpu().numpy()

    def zero(self) -> int:
        return 0

class DeepAccuracyMeter(AccuracyMeter):
    r"""Accuracy metter counting accuracy per level of hierarchy for all levels of hierarchy.
    """
    def __init__(self, hierarchy_size : Sequence[int], topk : int=None):
        super(DeepAccuracyMeter, self).__init__()
        if topk is None:
            self.topk=1
        else:
            self.topk=topk
        self.hierarchy_length=len(hierarchy_size)

    def __call__(self, input : Sequence[torch.Tensor], target: torch.Tensor) -> List:
        tab=[]
        for i in range(self.hierarchy_length):
            results=input[i]
            temp_topk=results.topk(self.topk,1)[1].squeeze(-1)
            if temp_topk.shape!=target[:,i].shape:
                temp_target=target[:,i,None]
            else:
                temp_target=target[:,i]
            tab.append(torch.sum(temp_topk==temp_target).cpu().numpy())
        return tab

    def zero(self) -> np.ndarray:
        return np.zeros((self.hierarchy_length))