from typing import Iterable, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

class MidOutBlock(nn.Module):
    r"""Blocks composed of two fully connected layers. Used to construct hierarchy-aware ResNets.
    Input dim is optionnal, in which case the layers will be initialised on the fly at first run.
    If using along torch.fx, consider running it on a dummy input to initialise all the layers before tracing through it.
    Same holds about torch_graph utilisation.
    """

    def __init__(self, mid_dim : int, out_dim : int, in_dim : Optional[int]=None):
        super(MidOutBlock, self).__init__()
        if in_dim is not None:
            self.initialised=True
            self.fc1=nn.Linear(in_dim,mid_dim)
        else:
            self.initialised=False
            self.mid_dim=mid_dim
        self.fc2=nn.Linear(mid_dim, out_dim)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        out=x.flatten(1)
        if not self.initialised:
            self.fc1=nn.Linear(out.shape[1],self.mid_dim)
            self.initialised=True
        out=self.fc1(out)
        out=F.leaky_relu(out)
        out=self.fc2(out)
        return out

class EndStack(nn.Module):
    r"""A stack of linear layers and leaky ReLUs to put at the end of a network to predict a hierarchy.
    Not used much in practice because it only extracts from the final activations.
    """

    def __init__(self, sizes : Sequence[int], output_ids : Iterable):
        super(EndStack, self).__init__()
        self.output_ids=set(output_ids) if hasattr(output_ids, "__iter__") else set([output_ids])
        self.endstack = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.length=len(self.endstack)

    def forward(self, x : torch.Tensor) -> List[torch.Tensor]:
        out=x.flatten(1)
        to_ret=[out]
        for i in range(self.length):
            out=self.endstack[i](out)
            if i in self.output_ids:
                to_ret.append(out)
            out=F.leaky_relu(out)
        return to_ret