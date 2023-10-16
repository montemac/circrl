"""Classes and functions to simplify the use of pytorch hooks for
arbitrary models, specifically for caching and patching activations, and
applying arbitrary hook functions safely."""

from typing import Optional, List, Dict, Callable
import torch as t
from torch import nn


class HookManager:
    """A context manager to simplify the use of pytorch hooks for
    arbitrary models, specifically for caching and patching activations,
    and applying arbitrary hook functions safely."""

    def __init__(
        self,
        model: nn.Module,
        cache: Optional[List[str]] = None,
        patch: Optional[Dict[str, t.Tensor]] = None,
        hook: Optional[Dict[str, Callable]] = None,
    ):
        """Initializes the hook manager with optional layers to cache,
        patch and/or hook. A layer may be specified in more than one
        category, in which case the order of operations is patch, hook,
        cache.  Patching occurs in-place and will broadcast as needed.

        Args:
            model: A model to manage hooks on
            cache: A list of strings representing the names of modules to
                cache activations for.
            patch: A dictionary mapping module names to tensors to patch
                activations with.
            hook: A dictionary mapping module names to hook functions to
                apply to activations. Hook function should take two
                arguments, input and output, and may modify output in
                place or return a new tensor.
        """
        self.model = model
        self.cache = cache
        self.patch = patch
        self.hook = hook
        self.handles = []
        self.cache_results: Dict[str, t.Tensor] = {}

    def __enter__(self):
        """Adds hooks to the model based on the previously specified
        cache, patch and hook attributes."""
        for name, module in self.model.named_modules():
            # The order here is important: it's allowable to catch,
            # patch and hook the same module, but the order of
            # operations is such that patches are applied first, then
            # any hooks, and finally cache hooks.  The forward hooks are
            # called in the order they are applied, so this ensures that
            # the cache hook stores the actual output of the chain of
            # hooks, and that the hook function is run after the patch
            # (though this final convention is somewhat arbitrary).
            if self.patch is not None and name in self.patch:
                handle = module.register_forward_hook(
                    self._make_patch_hook(name, self.patch[name])
                )
                self.handles.append(handle)
            if self.hook is not None and name in self.hook:
                handle = module.register_forward_hook(
                    self._make_func_hook(name, self.hook[name])
                )
                self.handles.append(handle)
            if self.cache is not None and name in self.cache:
                handle = module.register_forward_hook(
                    self._make_cache_hook(name)
                )
                self.handles.append(handle)
        return self.cache_results

    def __exit__(self, *args):
        """Removes all hooks from the model."""
        for handle in self.handles:
            handle.remove()

    def _make_cache_hook(self, name: str):
        """Returns a hook function that caches activations for a given
        module name."""

        # pylint: disable=unused-argument,redefined-builtin
        def hook(module, input, output):
            self.cache_results[name] = output

        return hook

    def _make_patch_hook(self, name: str, patch: t.Tensor):
        """Returns a hook function that patches activations for a given
        module name. Patching is done in-place using a copy operation,
        which will broadcase as needed based on the dimensions of the
        patch tensor."""

        # pylint: disable=unused-argument,redefined-builtin
        def hook(module, input, output):
            output.copy_(patch)

        return hook

    def _make_func_hook(self, name: str, func: Callable):
        """Returns a hook function that applies an arbitrary function to
        activations for a given module name."""

        # pylint: disable=unused-argument,redefined-builtin
        def hook(module, input, output):
            return func(input, output)

        return hook
