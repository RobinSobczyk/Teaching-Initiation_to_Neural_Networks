from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, Union
import torch
import networkx as nx
import torch.fx
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from torch.fx.node import map_arg, Target, Argument
from torch.fx.graph import PythonCode, _InsertPoint
from os import PathLike

def infun_node(fx_model : torch.fx.GraphModule, x : torch.fx.Node) -> str:
    r"""Used for classification of layers
    """
    if x.op=="placeholder":
        return("input")
    elif x.op=="get_attr":
        return("get_attr")
    elif x.op=="call_function":
        return(x.target.__name__)
    elif x.op=="call_module":
        return(type(reduce(lambda x,y: getattr(x,y), x.target.split('.'), fx_model)).__name__)
    elif x.op=="call_method":
        return(x.target)
    elif x.op=="output":
        return(x.op)

def maybe_add_labeled_edge(
    nx_graph : nx.MultiDiGraph,
    fx_child_node : torch.fx.Node,
    fx_parent_node : torch.fx.Node,
    edge_label : Optional[str]=None,
    shallow : bool=True,
    warning : bool=False
) -> None:
    r"""Adds a corresponding edge in ``nx_graph`` of one from``fx_parent_node`` to ``fx_child_node`` labeled with ``edge_label``
    If ``shallow`` is ``False``, then edge is labeled as deep.
    If ``Warning`` is ``True``, then warns when node creation is required in order to add edge.
    """
    if isinstance(fx_child_node, torch.fx.Node):
        if isinstance(fx_parent_node, torch.fx.Node):
            if not str(fx_child_node) in nx_graph:
                #ToDo implement verification through fx_child_node.graph
                nx_graph.add_node(str(fx_child_node))
                if warning:
                    warnings.warn(f"Node {str(fx_child_node)} was not already found in NetworkX graph and was added. torch.fx behaviour is undetermined.")
            if not str(fx_parent_node) in nx_graph:
                #ToDo implement verification through fx_parent_node.graph
                nx_graph.add_node(str(fx_parent_node))
                if warning:
                    warnings.warn(f"Node {str(fx_parent_node)} was not already found in NetworkX graph and was added. torch.fx behaviour is undetermined.")
            if shallow:
                nx_graph.add_edge(str(fx_parent_node),str(fx_child_node),key=edge_label,label=edge_label)
            else:
                id=nx_graph.add_edge(str(fx_parent_node),str(fx_child_node))
                nx_graph.edges[(str(fx_parent_node),str(fx_child_node),id)]["label"]=f"Deep {edge_label}\n edge id : {id}"
                nx_graph.deep_dict.setdefault((str(fx_child_node),edge_label),{}).setdefault(str(fx_parent_node),[]).append(id)
        else:
            raise TypeError("Input fx_parent_node is not of type torch.fx.Node")
    else:
        raise TypeError("Input fx_child_node is not of type torch.fx.Node")

def maybe_remove_edge(nx_graph : nx.MultiDiGraph, old_arg : Any, nx_child_node : str, label : str) -> None:
    r"""Remove all the edges in ``nx_graph`` corresponding to the ``label`` argument of torch.fx Node corresponding to ``nx_child_node``.
    ``old_arg`` is used in case it is of type torch.fx Node to delete the corresponding edge.
    """
    if (nx_child_node, label) in nx_graph.deep_dict:
        for arg_parent in nx_graph.deep_dict[(nx_child_node, label)]:
            for edge_id in nx_graph.deep_dict[(nx_child_node, label)][arg_parent]:
                nx_graph.remove_edge(arg_parent,nx_child_node,edge_id)
        del nx_graph.deep_dict[(nx_child_node, label)]
    if isinstance(old_arg, torch.fx.Node):
        nx_graph.remove_edge(str(old_arg),nx_child_node,label)

def filter_maybe_remove_replace_edge(
    nx_graph : nx.MultiDiGraph,
    old_arg : Any,
    nx_old_node : str,
    nx_child_node : str,
    label : str,
    default : Optional[Hashable]=None
) -> None:
    r"""Remove all edges in ``nx_graph`` corresponding to ``nx_old_node`` in the ``label`` argument of torch.fx Node corresponding to ``nx_child_node``.
    If ``default`` is not ``None``, then edges from ``nx_old_node`` to ``nx_child_node`` are converted to edges from ``default`` to ``nx_child_node``.
    """
    if (nx_child_node, label) in nx_graph.deep_dict:
        if nx_old_node in nx_graph.deep_dict[(nx_child_node, label)]:
            for edge_id in nx_graph.deep_dict[(nx_child_node, label)][nx_old_node]:
                nx_graph.remove_edge(nx_old_node,nx_child_node,edge_id)
                if default is not None:
                    id=nx_graph.add_edge(default,nx_child_node)
                    nx_graph.edges[(default,nx_child_node,id)]["label"]=f"Deep {label}\n edge id : {id}"
                    nx_graph.deep_dict.setdefault((nx_child_node,label),{}).setdefault(default,[]).append(id)
            del nx_graph.deep_dict[(nx_child_node, label)][nx_old_node]
    if isinstance(old_arg, torch.fx.Node) and str(old_arg)==nx_old_node:
        nx_graph.remove_edge(nx_old_node,nx_child_node,label)
        if default is not None:
            nx_graph.add_edge(default,nx_child_node,key=label,label=label)


def build_graph(fx_model : torch.fx.GraphModule) -> nx.MultiDiGraph:
    r"""Create a networkx.MultiDiGraph from a torch.fx.GraphModule
    """
    fx_graph=fx_model.graph
    nx_graph=nx.MultiDiGraph()
    nx_graph.dict={}
    nx_graph.deep_dict={}
    for x in fx_graph.nodes:
        node_type=infun_node(fx_model, x)
        if not str(x) in nx_graph:
            nx_graph.add_node(str(x), fx_node=x, node_type=node_type)
        else:
            nx_graph[str(x)]["fx_node"]=x
            nx_graph[str(x)]["node_type"]=node_type
        if node_type!="input":
            for i in range(len(x.args)):
                label=f"arg : {i}"
                shallow, inputs=detect_input(x.args[i])
                if shallow:
                    maybe_add_labeled_edge(nx_graph, x, inputs, edge_label=label, shallow=shallow)
                else:
                    for node in inputs:
                        maybe_add_labeled_edge(nx_graph, x, node, edge_label=label, shallow=shallow)
            for key in x.kwargs:
                label=f"kwarg : {key}"
                shallow, inputs=detect_input(x.kwargs[key])
                if shallow:
                    maybe_add_labeled_edge(nx_graph, x, inputs, edge_label=label, shallow=shallow)
                else:
                    for node in inputs:
                        maybe_add_labeled_edge(nx_graph, x, node, edge_label=label, shallow=shallow)
        if not node_type in nx_graph.dict:
            nx_graph.dict[node_type]=set()
        nx_graph.dict[node_type].add(str(x))
    return nx_graph

def detect_input(arg : Argument) -> Tuple[bool, Union[torch.fx.Node, List[torch.fx.Node]]]:
    r"""Return ``True, arg`` if ``arg`` is a torch.fx Node, else return ``False, node_list`` where ``node_list``is the list (with duplicity) of the torch.fx Nodes encountered in arg.
    Node search inside ``arg`` is done through ``torch.fx.node.map_arg``
    """
    if isinstance(arg,torch.fx.Node):
        return True, arg
    else:
        node_list=[]
        map_arg(arg, lambda x: node_list.append(x))
        return False, node_list

class BridgeGraph(object):
    r"""Used for representation of models as NetworkX graphs of layers and editing models through that representation.
    The class contain a torch.fx.GraphModule corresponding to the model loaded and a networkx.MultiDiGraph corresponding to the model.
    Note that any ``command`` method from torch.fx can be used on the torch.fx.GraphModule in the ``bridge_graph`` through ``bridge_graph.get_fx().command``.
    Loading a torch.fx.GraphModule directly from a NetworkX graph is currently not supported, and may never be.
    To update the subsequent torch.fx.GraphModule from NetworkX graph changes, changes should be made by updating step by step the NetworkX graph through the provided methods.
    Care that trace does stop at first ``return`` encountered and may not show the whole code of the input module, but rather only the executed code.
    """
    #ToDo support multiedge plot without superposition

    def __init__(self, model : Optional[torch.nn.Module]=None):
        r"""Create a BridgeGraph from the torch.nn.Module ``model``
        """
        if model is None:
            self.__fx_model=torch.fx.GraphModule()
            self.__nx_graph=nx.MultiDiGraph()
        else:
            self.__fx_model=torch.fx.symbolic_trace(model)
            self.__nx_graph=build_graph(self.__fx_model)

    def __getitem__(self, key : Hashable) -> Any:
        r"""Equivalent of ``get_nx()[key]``
        """
        return self.__nx_graph[key]

    def __call__(self, input : Any) -> Any:
        r"""Run the module corresponding to the torch.fx.GraphModule inside the BridgeGraph, on the given ``input``
        """
        return self.__fx_model(input)

    #get networkx and torch.fx parts

    def get_fx(self) -> torch.fx.GraphModule:
        r"""Return the torch.fx.GraphModule the BridgeGraph is working on
        """
        return self.__fx_model

    def get_nx(self) -> nx.MultiDiGraph:
        r"""Return the networkx.MultiDiGraph the BridgeGraph is working on
        """
        return self.__nx_graph

    @property
    def fx(self) -> torch.fx.GraphModule:
        r"""Return the torch.fx.GraphModule the BridgeGraph is working on
        """
        return self.get_fx()

    #useful methods on torch.fx.GraphModule

    def add_submodule(self, target : str, module : torch.nn.Module) -> bool:
        r"""Refer to torch.fx documentation, torch.fx.GraphModule method of same name
        """
        return self.__fx_model.add_submodule(target, module)

    def delete_all_unused_submodules(self) -> None:
        r"""Refer to torch.fx documentation, torch.fx.GraphModule method of same name
        """
        self.__fx_model.delete_all_unused_submodules()

    def delete_submodule(self,target : str) -> bool:
        r"""Refer to torch.fx documentation, torch.fx.GraphModule method of same name
        """
        return self.__fx_model.delete_submodule(target)

    def recompile(self) -> PythonCode:
        r"""Refer to torch.fx documentation, torch.fx.GraphModule method of same name
        """
        return self.__fx_model.recompile()

    def to_folder(self, folder : Union[str, PathLike], module_name : str ='FxModule') -> None:
        r"""Refer to torch.fx documentation, torch.fx.GraphModule method of same name
        """
        self.__fx_model.to_folder(folder, module_name)

    @property
    def graph(self) -> torch.fx.Graph:
        r"""Return the torch.fx.Graph corresponding to the GraphModule
        """
        return self.get_fx().graph

    #useful methods on torch.fx.Graph

    def lint(self) -> None:
        r"""Refer to torch.fx documentation, torch.fx.Graph method of same name
        """
        self.__fx_model.graph.lint()

    def eliminate_dead_code(self) -> bool:
        r"""Refer to torch.fx documentation, torch.fx.Graph method of same name
        """
        return self.__fx_model.graph.eliminate_dead_code()

    def inserting_after(self, node : Optional[torch.fx.Node]=None) -> Union[Any, _InsertPoint]:
        r"""Refer to torch.fx documentation, torch.fx.Graph method of same name
        """
        return self.__fx_model.graph.inserting_after(node)

    def inserting_before(self, node : Optional[torch.fx.Node]=None) -> Union[Any, _InsertPoint]:
        r"""Refer to torch.fx documentation, torch.fx.Graph method of same name
        """
        return self.__fx_model.graph.inserting_before(node)

    #getting torch.fx node from networkx name

    def get_fx_node(self, nx_node_name : str) -> torch.fx.Node :
        r"""return torch.fx Node corresponding to the node of name ``nx_node_name`` in the NetworkX graph
        """
        return self.__nx_graph.nodes.data()[nx_node_name]["fx_node"]

    #function to add/modify args/kwargs

    def add_arg_kwarg(self, arg : Argument , nx_node : str, position : Union[int, str]=0, mode : Optional[str]=None) -> None  :
        r"""If ``mode`` is ``"args"`` then set ``nx_node`` corresponding torch.fx Node args[``position``] to ``args`` and add corresponding edges.
        If args[``position``] is not valid, args are filled with ``None`` until reaching the right length.
        If args[``position``] is valid, delete the edges corresponding to the old args[``position``].
        If ``mode`` is ``"kwargs"`` then set ``nx_node`` corresponding torch.fx Node kwargs[``position``] to ``args`` and add corresponding edges.
        If ``position`` is not an existing key, it is added.
        If ``position`` already exists as a key, delete the edges corresponding to the old kwargs[``position``].
        If ```mode`` is ``None``, considered as ``"args"`` if ``position`` is of type ``int`` and ``"kwargs"`` otherwise.
        Aliased as ``add_deep_edge``.
        """
        if mode is None:
            if isinstance(position, int):
                mode="args"
            else:
                mode="kwargs"
        assert mode=="args" or mode=="kwargs", "Mode of execution is neither \"args\" nor \"kwargs\""
        fx_node=self.get_fx_node(nx_node)
        clean=False
        if mode=="args":
            assert isinstance(position, int), "position argument is not of type int"
            node_args=fx_node.args
            label=f"arg : {position}"
            if len(node_args)<=position:
                fx_node.args=node_args+(None,)*(position-len(node_args))+(arg,)
            else:
                old_arg=node_args[position]
                fx_node.update_arg(position,arg)
                clean=True
        else:
            label=f"kwarg : {position}"
            if position in fx_node.kwargs:
                old_arg=fx_node.kwargs[position]
                fx_node.update_kwarg(position,arg)
                clean=True
            else:
                new_kwargs=fx_node.kwargs.copy()
                new_kwargs[position]=arg
                fx_node.kwargs=new_kwargs
        if clean:
            maybe_remove_edge(self.__nx_graph, old_arg, nx_node, label)
        shallow, inputs=detect_input(arg)
        if shallow:
            maybe_add_labeled_edge(self.__nx_graph,fx_node,inputs,edge_label=label, shallow=shallow,warning=True)
        else:
            for node in inputs:
                maybe_add_labeled_edge(self.__nx_graph,fx_node,node,edge_label=label, shallow=shallow,warning=True)

    def add_arg(self, arg : Argument , nx_node : str, position : int=0) -> None:
        r"""Alias for ``add_arg_kwarg(self, arg, nx_node, position=position, mode="args")``
        """
        self.add_arg_kwarg(arg, nx_node, position=position, mode="args")

    def add_kwarg(self, arg : Argument , nx_node : str, key : str) -> None:
        r"""Alias for ``add_arg_kwarg(self, arg, nx_node, position=key, mode="kwargs")``
        """
        self.add_arg_kwarg(arg, nx_node, position=key, mode="kwargs")

    def set_args(self, args : Tuple[Argument,...], nx_node : str) -> None:
        r"""Set args of ``nx_node`` corresponding torch.fx Node to ``args`` and add corresponding edges.
        Delete all edges corresponding to old args of torch.fx Node
        """
        fx_node=self.get_fx_node(nx_node)
        node_args=fx_node.args
        for i in range(len(node_args)):
            label=f"arg : {i}"
            maybe_remove_edge(self.__nx_graph, node_args[i], nx_node, label)
        for i in range(len(args)):
            label=f"arg : {i}"
            shallow, inputs=detect_input(args[i])
            if shallow:
                maybe_add_labeled_edge(self.__nx_graph, fx_node, inputs, edge_label=label, shallow=shallow)
            else:
                for node in inputs:
                    maybe_add_labeled_edge(self.__nx_graph, fx_node, node, edge_label=label, shallow=shallow)
        fx_node.args=args

    def set_kwargs(self, kwargs : Dict[str, Argument], nx_node) -> None:
        r"""Set kwargs of ``nx_node`` corresponding torch.fx Node to ``kwargs`` and add corresponding edges.
        Delete all edges corresponding to old kwargs of torch.fx Node
        """
        fx_node=self.get_fx_node(nx_node)
        node_kwargs=fx_node.kwargs
        for key in node_kwargs:
            label=f"kwarg : {key}"
            maybe_remove_edge(self.__nx_graph, node_kwargs[key], nx_node, label)
        for key in kwargs:
            label=f"kwarg : {key}"
            shallow, inputs=detect_input(kwargs[key])
            if shallow:
                maybe_add_labeled_edge(self.__nx_graph, fx_node, inputs, edge_label=label, shallow=shallow)
            else:
                for node in inputs:
                    maybe_add_labeled_edge(self.__nx_graph, fx_node, node, edge_label=label, shallow=shallow)
        fx_node.kwargs=kwargs


    #adapted networkx methods

    def add_node(
        self,
        name : str,
        op : str,
        target : Target,
        args : Optional[Tuple[Argument,...]],
        kwargs : Optional[Dict[str, Argument]],
        return_type : Optional[Any]=None
    ) -> str:
        r"""Will create a torch.fx Node in the torch.fx.GraphModule. Be sure that the torch.fx Node you are going to create is valid. Refer to torch.fx documentation for more informations.
        Create a corresponding NetworkX node and add the edges contained in ``args`` and ``kwargs``.
        Return the name under which the NetworkX node is registered.
        """
        new_node=self.graph.create_node(op, target, args, kwargs, name, return_type)
        node_type=infun_node(self.__fx_model,new_node)
        self.__nx_graph.add_node(str(new_node), fx_node=new_node, node_type=node_type)
        if node_type!="input":
            for i in range(len(new_node.args)):
                maybe_add_labeled_edge(self.__nx_graph, new_node, new_node.args[i], edge_label=f"arg : {i}", warning=True)
            for key in new_node.kwargs:
                maybe_add_labeled_edge(self.__nx_graph, new_node, new_node.kwargs[key], edge_label=f"kwarg : {key}", warning=True)
        if not node_type in self.__nx_graph.dict:
            self.__nx_graph.dict[node_type]=set()
        self.__nx_graph.dict[node_type].add(str(new_node))
        return str(new_node)

    def add_edge(self, nx_parent_node : str, nx_child_node : str, position : Union[int, str]=0, mode : Optional[str]=None) -> None:
        r"""If ``mode`` is ``"args"`` then set ``nx_node`` corresponding torch.fx Node args[``position``] to torch.fx Node corresponding to ``nx_parent_node`` and add corresponding edges.
        If args[``position``] is not valid, args are filled with ``None`` until reaching the right length.
        If args[``position``] is valid, delete the edges corresponding to the old args[``position``].
        If ``mode`` is ``"kwargs"`` then set ``nx_node`` corresponding torch.fx Node kwargs[``position``] to torch.fx Node corresponding to ``nx_parent_node`` and add corresponding edges.
        If ``position`` is not an existing key, it is added.
        If ``position`` already exists as a key, delete the edges corresponding to the old kwargs[``position``].
        If ```mode`` is ``None``, considered as ``"args"`` if ``position`` is of type ``int`` and ``"kwargs"`` otherwise.
        """
        if mode is None:
            if isinstance(position, int):
                mode="args"
            else:
                mode="kwargs"
        assert mode=="args" or mode=="kwargs", "Mode of execution is neither \"args\" nor \"kwargs\""
        fx_parent_node=self.get_fx_node(nx_parent_node)
        fx_child_node=self.get_fx_node(nx_child_node)
        clean=False
        if mode=="args":
            assert isinstance(position, int), "position argument is not of type int"
            child_args=fx_child_node.args
            label=f"arg : {position}"
            if len(child_args)<=position:
                fx_child_node.args=child_args+(None,)*(position-len(child_args))+(fx_parent_node,)
            else:
                old_arg=child_args[position]
                fx_child_node.update_arg(position,fx_parent_node)
                clean=True
        else:
            label=f"kwarg : {position}"
            child_kwargs=fx_child_node.kwargs
            if position in child_kwargs:
                old_arg=child_kwargs[position]
                fx_child_node.update_kwarg(position,fx_parent_node)
                clean=True
            else:
                new_kwargs=fx_child_node.kwargs.copy()
                new_kwargs[position]=fx_parent_node
                fx_child_node.kwargs=new_kwargs
        if clean:
            maybe_remove_edge(self.__nx_graph, old_arg, nx_child_node, label)
        self.__nx_graph.add_edge(nx_parent_node,nx_child_node, key=label, label=label)

    add_deep_edge = add_arg_kwarg

    def del_node(self, nx_node : str) -> None:
        r"""Delete corresponding torch.fx Node from torch.fx.GraphModule, replacing it with ``None`` where needed.
        Delete "nx_node" and all the corresponding edges in the NetworkX graph
        """
        fx_node=self.get_fx_node(nx_node)
        users=fx_node.users.copy()
        for fx_child in users:
            fx_child.replace_input_with(fx_node, None)
        args=fx_node.args
        kwargs=fx_node.kwargs
        for i in range(len(args)):
            label=f"arg : {i}"
            maybe_remove_edge(self.__nx_graph, args[i], nx_node, label)
        for key in kwargs:
            label=f"kwarg : {key}"
            maybe_remove_edge(self.__nx_graph, kwargs[key], nx_node, label)
        self.__fx_model.graph.erase_node(fx_node)
        node_type=self.__nx_graph.nodes.data()[nx_node]["node_type"]
        self.__nx_graph.dict[node_type].discard(nx_node)
        self.__nx_graph.remove_node(nx_node)

    def del_edges(
        self,
        nx_parent_node : str,
        nx_child_node : str,
        position : Optional[Union[int, str]]=None,
        mode : Optional[str]=None,
        default : Argument =None
    ) -> None:
        r"""Delete corresponding edges of those created from ``mode[position]`` in torch.fx.GraphModule, replacing them with ``default`` if of type torch.fx Node.
        Delete corresponding edges in NetworkX graph, creating edges between ``default`` and ``nx_child_node`` if ``default`` of type torch.fx Node.
        Adds ``default`` to NetworkX graph if needed BUT NOT IN THE torch.fx.GraphModule !
        If ``position`` is ``None``, goes through all arguments in ``mode``. If ``mode`` is ``None``, goes through both args and kwargs.
        """
        to_pass_default=None
        if isinstance(default, torch.fx.Node):
            if not str(default) in self.__nx_graph:
                #ToDo implement verification through default.graph
                self.__nx_graph.add(str(default))
                warnings.warn(f"Node {str(default)} was not already found in NetworkX graph and was added. torch.fx behaviour is undetermined.")
            to_pass_default=str(default)
        fx_parent_node=self.get_fx_node(nx_parent_node)
        fx_child_node=self.get_fx_node(nx_child_node)
        if position is None:
            assert mode=="args" or mode=="kwargs" or mode is None, "Mode of execution is not \"args\", \"kwargs\" or None"
            if mode=="args" or mode is None:
                child_args=fx_child_node.args
                for i in range(len(child_args)):
                    label=f"arg : {i}"
                    filter_maybe_remove_replace_edge(self.__nx_graph, child_args[i], nx_parent_node, nx_child_node, label, default=to_pass_default)
                new_args=map_arg(child_args, lambda i : default if i==fx_parent_node else i)
                fx_child_node.args=new_args
            if mode=="kwargs" or mode is None:
                child_kwargs=fx_child_node.kwargs.copy()
                for key in child_kwargs:
                    label=f"kwarg : {key}"
                    filter_maybe_remove_replace_edge(self.__nx_graph, child_kwargs[key], nx_parent_node, nx_child_node, label, default=to_pass_default)
                new_kwargs=map_arg(child_kwargs, lambda i : default if i==fx_parent_node else i)
                fx_child_node.kwargs=new_kwargs
        else:
            assert mode=="args" or mode=="kwargs", "Mode of execution is neither \"args\" nor \"kwargs\" but a positionnal argument was passed"
            if mode=="args":
                child_arg=fx_child_node.args[position]
                label=f"arg : {position}"
                filter_maybe_remove_replace_edge(self.__nx_graph, child_arg, nx_parent_node, nx_child_node, label, default=to_pass_default)
                new_arg=map_arg(child_arg, lambda i : default if i==fx_parent_node else i)
                fx_child_node.update_arg(position,new_arg)
            else:
                child_kwarg=fx_child_node.kwargs[position]
                label=f"kwarg : {position}"
                filter_maybe_remove_replace_edge(self.__nx_graph, child_kwarg, nx_parent_node, nx_child_node, label, default=to_pass_default)
                new_kwarg=map_arg(child_kwarg, lambda i : default if i==fx_parent_node else i)
                fx_child_node.update_kwarg(position,new_kwarg)

    #update networkx part from torch.fx part, useful after some custom commands like ``bridge_graph.get_fx().command``, recompile may be needed

    def update_nx(self) -> None:
        r"""Rebuild the NetworkX graph from the subsequent torch.fx.GraphModule.
        """
        self.__nx_graph=build_graph(self.__fx_model)

    #function to unfold nested modules

    def retrace(self) -> None:
        r"""If used properly, should allow to unnest nested modules and functions.
        """
        self.__fx_model=torch.fx.symbolic_trace(self.__fx_model)
        self.update_nx()

    #plotting networkx part

    def plot(self, x : float, y : float) -> None:
        r"""Plot the NetworkX graph with matplotlib.
        Does not currently support parallel edges.
        """
        #ToDo add support for multiedges
        plt.figure(figsize=(x,y))
        pos=nx.nx_agraph.pygraphviz_layout(self.__nx_graph, prog="dot")
        colors=plt.cm.get_cmap('hsv', len(self.__nx_graph.dict)+1)
        i=0
        handles=[]
        for x in self.__nx_graph.dict:
            nx.draw_networkx_nodes(self.__nx_graph, pos=pos, nodelist=self.__nx_graph.dict[x], node_color=np.array([colors(i)]))
            patch = mpatches.Patch(color=colors(i), label=x)
            handles.append(patch)
            i+=1
        plt.legend(handles=handles)
        plt.autoscale(enable=True)
        nx.draw_networkx_edges(self.__nx_graph, pos)
        label_dict={}
        for u,v,data in self.__nx_graph.edges(data=True):
            if "label" in data:
                label_dict[(u,v)]=data["label"]
        nx.draw_networkx_edge_labels(self.__nx_graph, pos, edge_labels=label_dict, rotate=False)
        nx.draw_networkx_labels(self.__nx_graph, pos)
        plt.show()