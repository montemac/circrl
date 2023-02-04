import copy
from contextlib import contextmanager
from dataclasses import dataclass
import typing
from typing import Dict
import warnings

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import xarray as xr
from einops import rearrange
from bidict import bidict

# TODO: simplify dependencies for visualizing computational graph
# import panel as pn
# pn.extension('plotly', comms='ipywidgets')
# import hvplot.networkx as hvnx
# import networkx as nx
# from networkx.algorithms import bipartite
# import holoviews as hv

@dataclass
class ModuleHookData():
    inputs:      tuple
    output:      t.Tensor
    custom_data: typing.Any

@dataclass
class PatchDef():
    '''Both tensors should be the size of the activation value to be patched, 
    with the first batch dimension optionally length-1 to broadcast in this dim.'''
    mask:  t.Tensor  # True where patched values should be applied
    value: t.Tensor  # Patch values

class ModuleHook():
    network: nn.Module
    values_by_label: bidict[str, t.Tensor]
    modules_by_label: Dict[str, nn.Module]
    module_labels_by_id: Dict[int, str]
    module_data_by_label: Dict[str, typing.Any]
    meta_by_label: Dict[str, typing.Any]
    value_to_module_edges: set[tuple[str, str]]
    module_to_value_edges: set[tuple[str, str]]

    def __init__(self, network, module_labels=None):
        '''Copy the provided network and add forward hooks to extract values from all layers
        during subsequent forward calls.  Also creates data structures to hold module input/output
        values and relationships to allow building of the module graph.'''
        # Copy network
        self.network = copy.deepcopy(network)
        # Init data structures
        self.modules_by_label = {}
        self.module_labels_by_id = {}
        self.module_data_by_label = {}
        self.meta_by_label = {}
        self.values_by_label = bidict()
        self.value_to_module_edges = set()
        self.module_to_value_edges = set()
        self.module_data_by_label = {}
        # Register all the modules with labels
        if module_labels is not None:
            for label, mod in zip(module_labels, self.network.modules()):
                self.register_module(mod, label)
        else:
            # Create module labels based on internal module variable names if not provided
            for label, mod in self.network.named_modules():
                self.register_module(mod, label)
        # Attach the forward hook function to all modules in the provided network
        self.network.apply(lambda m: self.register_hook(m))
        # Init some variables
        self.hook_should_get_custom_data = False
        self.batch_index = None
        self.last_batch_index = None
        self.patches = {}

    def register_hook(self, module):
        # Register forward hook on provided module
        module.register_forward_hook(self._module_hook)

    @contextmanager
    def set_hook_should_get_custom_data(self):
        self.hook_should_get_custom_data = True
        yield None
        self.hook_should_get_custom_data = False

    @contextmanager
    def use_patches(self, patches):
        self.patches = patches
        yield None
        self.patches = {}

    @contextmanager
    def use_batch_index(self, batch_index):
        self.batch_index = batch_index
        yield None
        self.batch_index = None

    def tensor_to_result_object(self, tensor, label):
        array = tensor.detach().numpy()
        if self.last_batch_index is None:
            return array
        dims = [self.last_batch_index.name] + ['{}_d{}'.format(label, ii) for ii in range(1, len(array.shape))]
        coords = {dims[0]: self.last_batch_index}
        for dim, ln in list(zip(dims, array.shape))[1:]:
            coords[dim] = np.arange(ln)
        return xr.DataArray(array, dims=dims, coords=coords)

    def register_module(self, module, label):
        '''If a module isn't registered already, add it to the various data structures.'''
        _id = id(module)
        if _id not in self.module_labels_by_id:
            self.module_labels_by_id[_id] = label
            self.modules_by_label[label] = module
            self.meta_by_label[label] = dict(desc=' '.join(str(module).split()))
        return self.module_labels_by_id[_id]

    def set_module_data(self, label, module_data):
        self.module_data_by_label[label] = module_data

    def set_value(self, label: str, value: t.Tensor):
        # self.values_by_label is a bidict, so keys (labels) and values (value
        # tensors) are both unique sets.
        # First, check if this value object is already in the bidict; if so,
        # return it's label (value labels shouldn't ever change)
        if value in self.values_by_label.inverse:
            return self.values_by_label.inverse[value]
        # Otherwise, set the provided label to point to this value
        self.values_by_label[label] = value
        try:
            desc = str(value.shape) # Value is a single tensor
        except AttributeError:
            # Value might be a tuple of tensors?
            try:
                desc = '({})'.format(', '.join([str(vv.shape) for vv in value]))
            except AttributeError:
                desc = ''
        self.meta_by_label[label] = dict(desc=desc)
        return label

    def _module_hook(self, module, inps, outp):
        # Store the batch index that should apply to current values
        self.last_batch_index = self.batch_index
        # Get the label for this module
        module_label = self.module_labels_by_id[id(module)]
        # Store the input values
        inp_labels = []
        for ii, inp in enumerate(inps):
            inp_labels.append(self.set_value("{}_in{}".format(module_label, ii), inp))
        # Store the output value
        outp_label = self.set_value("{}_out".format(module_label), outp)
        # Store edges to this module
        for inp_label in inp_labels:
            self.value_to_module_edges.add((inp_label, module_label))
        # Store output edge from this module, but only if the output object is 
        # different to all input objects (i.e. avoid cycles if we have identity modules)
        if not id(outp) in [id(inp) for inp in inps]:
            self.module_to_value_edges.add((module_label, outp_label))        

        # Populate custom data, if enabled (this can be large)
        if self.hook_should_get_custom_data:
            # Create module data object
            # Convert input and output objects to appropriate type depending on
            # whether an index is set
            module_data = ModuleHookData(inps, outp, None)
            # Add custom data for this module based on module type
            if isinstance(module, nn.Linear):
                module_data.custom_data = dict(weight=module.state_dict()['weight'],
                    bias=module.state_dict()['bias'])
                if inps[0].shape[0] == 1:
                    # Include pre-sum data if we're only looking at a single input point,
                    # otherwise could use too much memory?
                    presum = inps[0] * module.state_dict()['weight']
                    module_data.custom_data['presum'] = presum
            if isinstance(module, nn.Conv2d):
                weight = module.state_dict()['weight']
                module_data.custom_data = dict(weight=weight,
                    bias=module.state_dict()['bias'])
                if inps[0].shape[0] == 1:
                    # Include pre-sum data if we're only looking at a single input point,
                    # otherwise could use too much memory?
                    # Pre-calculate the by-channel convolutions before the final sums
                    # Weight tensor dims are (out_channel, in_channel, kh, kw)
                    # We can calculate the intermediate presum values by running a single
                    # conv2d call for each input, but across all outputs
                    # TODO: handle groups argument
                    # TODO: consider making all custom data (e.g. presums, etc. xarrays to
                    # make plotting easier)
                    assert module.groups == 1, 'Function only supports groups=1 for now'
                    presum_list = []
                    for inch in range(weight.shape[1]):
                        presum_this_input = nn.functional.conv2d(
                            inps[0][:,inch:(inch+1),:,:], weight=weight[:,inch:(inch+1),:,:], 
                            stride=module.stride, padding=module.padding, 
                            dilation=module.dilation)
                        presum_list.append(presum_this_input)
                    presum_tensor = rearrange(presum_list, "inch b outch h w -> inch outch b h w")
                    module_data.custom_data['presum'] = presum_tensor
            # Store the module data
            self.set_module_data(module_label, module_data)
            
        # Handle patching, if any
        if outp_label in self.patches:
            patch = self.patches[outp_label]
            if isinstance(patch, PatchDef):
                pmask = patch.mask
                pvalue = patch.value
                outp = outp*(~pmask) + pvalue*pmask
            else:
                # Patch must be a custom function
                outp = patch(outp)
            #for slice_tuple, patched_value in self.patches[outp_label]:
            #    outp[slice_tuple] = t.Tensor(patched_value)
            # Update the stored value since it's changed
            self.set_value("{}_out".format(module_label), outp)
            return outp

    def probe_with_input(self, inp, patches={}, func=None, **kwargs):
        warnings.warn('''Function probe_with_input is deprecated due to
        a confusing name and has been replaced with run_with_input.''')
        return self.run_with_input(inp, patches, func, **kwargs)
    
    def run_with_input(self, inp, patches={}, func=None, **kwargs):
        '''Makes a forward pass over the network, creating all intermediate module
        input/output values.  By default, customizes the forward call to use predict() 
        if module type supports it, so that we get all the observation pre-processing,
        etc, otherwise uses a normal forward call.  The function to be called to
        execute the forward pass can also be overriden using the func argument,
        which will be called with the hooked network, input as a tensor and **kwargs.
        Additional keyword arguments will be passed to the forward call in all cases.'''
        # Make sure the input is an ndarray at this stage, but if DataArray is passed,
        # all hook results will also be DataArrays with same batch dimension (assumed
        # to be first dimension implicitly here), else numpy ndarrays.
        batch_index = None
        if isinstance(inp, xr.DataArray):
            batch_index = inp.indexes[inp.dims[0]]
            inp = inp.to_numpy()
        # Make forward pass using patches, batch index (if any), and storing intermediate data
        with self.set_hook_should_get_custom_data(), self.use_patches(patches), \
                self.use_batch_index(batch_index), t.no_grad():
            # If a function is provided, call it with the input as a tensor
            if func is not None:
                outp = func(self.network, t.from_numpy(inp), **kwargs)
            else:
                # Default to using predict() if present, otherwise normal forward call
                try:
                    # predict() wants an ndarray
                    outp = self.network.predict(inp, **kwargs)
                except AttributeError:
                    # Normal module forward call wants a tensor
                    outp = self.network(t.from_numpy(inp), **kwargs)
        return outp        
            

    def get_module_and_data_by_label(self, label):
        module = self.modules_by_label[label]
        data = self.module_data_by_label[label]
        return module, data

    def get_value_by_label(self, label, convert=True):
        value = self.values_by_label[label]
        if convert:
            return self.tensor_to_result_object(value, label)
        return value

    def get_preceders(self, label, include_parent_modules=False):
        '''Step backward through computational graph and return preceding
        module/value label.'''
        # TODO: make this more efficient, prob shouldn't need to loop through edges?
        preceders = []
        for n1, n2 in (list(self.module_to_value_edges) + list(self.value_to_module_edges)):
            if n2 == label:
                preceders.append(n1)
        if not include_parent_modules:
            preceders = [prec for prec in preceders if not (prec in self.modules_by_label and
                len(list(self.modules_by_label[prec].children()))!=0)]
        return preceders

    # TODO: simplify dependencies
    # def get_graph_after_probe(self, include_parent_modules=True, show_unconnected=False):
    #     # Nodes and edges using labels
    #     def make_node_tuples(labels):
    #         return [(label, self.meta_by_label[label]) for label in labels]
    #     value_nodes = make_node_tuples(self.values_by_label.keys())
    #     modules_with_edges = set([mod for val, mod in self.value_to_module_edges] +
    #         [mod for mod, val in self.module_to_value_edges])
    #     modules_to_use = set([label for label, mod in self.modules_by_label.items() 
    #         if ((include_parent_modules or len(list(mod.children()))==0) and
    #             (show_unconnected       or label in modules_with_edges)     )])
    #     module_nodes = make_node_tuples(modules_to_use)
    #     edges = [(mod, val) for mod, val in self.module_to_value_edges if mod in modules_to_use] + \
    #         [(val, mod) for val, mod in self.value_to_module_edges if mod in modules_to_use]
    #     # Build the graph
    #     B = nx.DiGraph()
    #     B.add_nodes_from(module_nodes, type='module')
    #     B.add_nodes_from(value_nodes, type='value')
    #     B.add_edges_from(edges)
    #     # Lay out the graph
    #     pos = nx.layout.kamada_kawai_layout(B)
    #     # Calculate node colors 
    #     node_top_level_by_label = {lab: ii for ii, lab in enumerate(list(nx.topological_sort(B)))}
    #     node_top_levels = [node_top_level_by_label[lab] for lab in B.nodes]
    #     # Create the plot
    #     viz = hvnx.draw(B, pos, node_color=node_top_levels, cmap='Blues', 
    #         node_size=200-(hv.dim('type')=='value')*100, arrowhead_length=0.01 ,width=800, height=800)
    #     # Add the responsive tab object
    #     stream_selection = hv.streams.Selection1D(source=viz.nodes)
    #     @pn.depends(stream_selection.param.index)
    #     def selection(index):
    #         if len(index) == 0:
    #             ss = 'No selection'
    #         else:
    #             ii = index[0]
    #             label = list(B.nodes)[ii]
    #             obj = self.objects_by_label[label]
    #             _id = id(obj)
    #             ss = '\n'.join([
    #                 'Node index: {}'.format(ii),
    #                 'Label: {}'.format(label),
    #                 'Type: {}'.format('module' if isinstance(obj, nn.Module) else 'value'),
    #                 'ID: {}'.format(_id)])
    #         return pn.pane.Str(ss, width=200)
    #     #return pn.Column(viz.nodes.opts(alpha=0) * viz, selection)
    #     return pn.Pane(viz)
