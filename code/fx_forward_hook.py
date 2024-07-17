import torch.fx

class FxForwardHook(object):
    r"""A try at implementing (forward hook)-like objects in `torch.fx`.
    This implementation is not fully functionnal and is only developped so far for feature extraction.
    """

    def __init__(self):
        pass

    def __call__(self,*args, **kwargs):
        raise NotImplementedError

    def __fx_create_arg__(self, proxy : torch.fx.Tracer) -> torch.fx.Node:
        if not(hasattr(proxy, f'{self.__name__}_attrs')):
            setattr(proxy, f'{self.__name__}_attrs', {})
        qualname = getattr(proxy, f'{self.__name__}_attrs').get(self)
        if not qualname:
            i=0
            while True:
                qualname = f'_{self.__name__}_object{i}'
                if not hasattr(proxy.root, qualname):
                    break
                i += 1
            getattr(proxy, f'{self.__name__}_attrs')[self]=qualname
            setattr(proxy.root, qualname, self)

        return proxy.create_node('get_attr', qualname, (), {})

@torch.fx.wrap
def wrapped_getattr(instance, method, *args, **kwargs):
    r"""getattr method that is wrapped in order to call hooks, trace, retrace etc without problems
    """
    return getattr(instance, method)(*args, **kwargs)

def fx_forward_hook_register(module : torch.fx.GraphModule, hook : FxForwardHook, default : bool =True) -> torch.fx.GraphModule:
    r"""Call `hook` after every node where `to_be_hooked` attribute is set to `True` on the output of the node.
    If `to_be_hooked` attribute cannot be found, `default` value is used instead.
    `hook` is registered inside `_fxfh` attribute of `module`
    `module` must be a `torch.fx.GraphModule`
    """
    setattr(module,"_fxfh", hook)
    with module.graph.inserting_before():
        hook_node=module.graph.get_attr("_fxfh")
        hook_node.to_be_hooked=False
    for node in module.graph.nodes:
        if getattr(node, "to_be_hooked", default):
            with module.graph.inserting_after(node):
                hook_call_node=module.graph.call_function(wrapped_getattr)
            hook_call_node.to_be_hooked=False
            node.replace_all_uses_with(hook_call_node)
            hook_call_node.args=(hook_node, "__call__", node)
    module.recompile()
    return module

def fx_forward_hook_remove(module : torch.fx.GraphModule) -> torch.fx.GraphModule:
    r"""Remove hooks from a `module` of type `torch.fx.GraphModule`
    """
    for node in module.graph.nodes:
        if node.op=="get_attr" and node.target=="_fxfh":
            hook_node=node
            for hook_call_node in hook_node.users:
                if hook_call_node.op=="call_function" and hook_call_node.target==wrapped_getattr:
                    parent_node=hook_call_node.args[2]
                    hook_call_node.replace_all_uses_with(parent_node)
                    module.graph.erase_node(hook_call_node)
                else:
                    raise NotImplementedError("Node format unsupported")
            module.graph.erase_node(hook_node)
            delattr(module,"_fxfh")
    module.recompile()
    return module