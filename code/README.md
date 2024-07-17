Quick map of the code :  
core modules :
- `torch_graph.py` : module for NN tracing and interaction through NetworkX
- `fx_forward_hook.py` : attempt of forward hooks for torch graphs, that are resilient to recompile

Things to rework or  here for examples :
- `CenterLoss.py` : an example of implementation of a custom loss
- `MyLosses.py` : contains some losses as well as accuracy meters
- `MyTrain.py` : training loop and grad tracking utilities
- `MyModules.py` : custom modules
- `CifarResNetGen.py` : utilities for autogeneration of Cifar ResNets