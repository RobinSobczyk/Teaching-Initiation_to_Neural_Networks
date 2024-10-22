Quick map of the code :  
core modules :
- `torch_graph.py` : module for NN tracing and interaction through NetworkX
- `fx_forward_hook.py` : attempt of forward hooks for torch graphs, that are resilient to recompile
- `network_gen.py` : utilities for autogeneration of neural networks, currrently include Cifar ResNets